import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from itertools import cycle
import models.transformer as transformer
import models.StyTR as StyTR
from torchvision.utils import save_image
import collections

# Funktion zur Definition der Transformationen, die auf die Bilder angewendet werden
# Die Bilder werden auf die Größe (256, 256) skaliert und in Tensors konvertiert
def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

# Dataset-Klasse zum Laden von Content- und Ground-Truth-Bildern
class ExtendedPairedDataset(data.Dataset):
    def __init__(self, content_root, gt_root, transform, use_color_input=False, color_value=None):
        super(ExtendedPairedDataset, self).__init__()
        self.content_root = content_root
        self.gt_root = gt_root
        self.transform = transform
        self.use_color_input = use_color_input
        self.color_value = color_value

        # Lade und sortiere die Content-Bilder alphabetisch
        if not os.path.exists(content_root):
            raise FileNotFoundError(f"Content directory '{content_root}' does not exist.")
        self.content_paths = sorted([os.path.join(content_root, f) for f in os.listdir(content_root)])

        # Lade und sortiere die Ground Truth-Bilder aus dem Verzeichnis
        if not os.path.exists(gt_root):
            raise FileNotFoundError(f"Ground Truth directory '{gt_root}' does not exist.")
        self.gt_paths = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)])

        # Überprüfen, ob die Ordner die gleiche Anzahl von Bildern haben
        if len(self.content_paths) != len(self.gt_paths):
            raise ValueError("Content and Ground Truth folders must contain the same number of images.")

    def __getitem__(self, index):
        # Lade Content- und Ground Truth-Bilder anhand des Index
        content_path = self.content_paths[index]
        gt_path = self.gt_paths[index]

        # Lade und transformiere das Content- und Ground Truth-Bild
        content_img = Image.open(content_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        content_img = self.transform(content_img)
        gt_img = self.transform(gt_img)

        if self.use_color_input:
            # Erzeuge ein RGB-Bild aus der angegebenen Farbe, falls Farbinformation verwendet wird
            style_img = torch.ones((3, 256, 256)) * self.color_value
        else:
            style_img = None  # Kein explizites Style-Bild notwendig

        return content_img, style_img, gt_img

    def __len__(self):
        return len(self.content_paths)

# Funktion zur Anpassung der Lernrate während des Trainings
# Die Lernrate sinkt nach einer bestimmten Anzahl von Iterationen
def adjust_learning_rate(optimizer, iteration_count):
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Funktion zum "Aufwärmen" der Lernrate zu Beginn des Trainings
def warmup_learning_rate(optimizer, iteration_count):
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Argumente für den Trainingsprozess definieren
parser = argparse.ArgumentParser()
parser.add_argument('--content_dir', default='/bigwork/nhg0fenn/StyTr²FEM/images/content/CT', type=str, help='Directory path to a batch of content images')
parser.add_argument('--stretch_level', type=str, choices=['FEM1', 'FEM4'], required=True, help="Stretch level to apply: FEM1 (1% stretch, green) or FEM4 (4% stretch, red)")
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--save_dir', default='./experiments', help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs', help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--gt_weight', type=float, default=5.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int, help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--use_color_input', action='store_true', help='Use a solid color as style input instead of a style image')
try:
    args = parser.parse_args()
except SystemExit as e:
    raise ValueError("Error in command line arguments. Please check your inputs.")

# Bestimme den Ground Truth-Ordner und die Farbe basierend auf dem Stretch-Level
if args.stretch_level == 'FEM1':
    gt_dir = '/bigwork/nhg0fenn/StyTr²FEM/images/style/FEM1'
    color_value = torch.tensor([0.0, 1.0, 0.0]).view(3, 1, 1)  # Grün für 1% Streckung
elif args.stretch_level == 'FEM4':
    gt_dir = '/bigwork/nhg0fenn/StyTr²FEM/images/style/FEM4'
    color_value = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)  # Rot für 4% Streckung
else:
    raise ValueError("Stretch level must be either 'FEM1' or 'FEM4'.")

# Prüfen, ob CUDA (GPU) verfügbar ist
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

# Erstelle Verzeichnisse, falls sie nicht vorhanden sind
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

# Initialisiere das VGG-Netzwerk, den Decoder und die Transformer-Komponenten
vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()
Trans = transformer.Transformer()

# Initialisiere das Netzwerk ohne Gradientenberechnung
with torch.no_grad():
    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
network.train()
network.to(device)

# Falls mehrere GPUs verfügbar sind, verwende DataParallel
if torch.cuda.device_count() > 1:
    network = nn.DataParallel(network, device_ids=list(range(torch.cuda.device_count())))
    use_data_parallel = True
else:
    use_data_parallel = False

# Transformation für die Trainingsbilder
transform = train_transform()

# Erstelle das Dataset mit Content- und Ground Truth-Paaren
paired_dataset = ExtendedPairedDataset(
    args.content_dir,
    gt_dir,
    transform,
    use_color_input=args.use_color_input,
    color_value=color_value
)

# DataLoader für das Dataset
paired_loader = data.DataLoader(
    paired_dataset, batch_size=args.batch_size,
    shuffle=False,  # Shuffle ist deaktiviert, um die Reihenfolge der Bilder beizubehalten
    num_workers=args.n_threads
)


# Endloser Iterator für den DataLoader
paired_iter = cycle(paired_loader)

# Verlustfunktion für die Ground Truth-Bilder
criterion_gt = nn.MSELoss()

# Optimierer definieren
optimizer = torch.optim.Adam(
    [{'params': network.module.transformer.parameters()} if use_data_parallel else {'params': network.transformer.parameters()},
     {'params': network.module.decode.parameters()} if use_data_parallel else {'params': network.decode.parameters()},
     {'params': network.module.embedding.parameters()} if use_data_parallel else {'params': network.embedding.parameters()}],
    lr=args.lr
)

# Initialisiere eine Deque, um die letzten 50 Verluste zu speichern
last_50_losses = collections.deque(maxlen=50)

# Verzeichnis für die Ausgabe der Vergleichsbilder erstellen
output_dir = os.path.join(args.save_dir, "test")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Trainingsschleife
for i in tqdm(range(args.max_iter)):
    if i < 1e4:
        # Während der ersten Iterationen wird die Lernrate aufgewärmt
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        # Danach wird die Lernrate kontinuierlich angepasst
        adjust_learning_rate(optimizer, iteration_count=i)

    # Lade die nächsten Content- und Ground Truth-Bilder aus dem endlosen Iterator
    content_images, _, gt_images = next(paired_iter)
    content_images, gt_images = content_images.to(device), gt_images.to(device)

    # Style Transfer und Verlustberechnung
    out, loss_c, _, l_identity1, l_identity2 = network(content_images, content_images)

    # Ground Truth-Verlust berechnen
    loss_gt = criterion_gt(out, gt_images) * args.gt_weight

    # Verlustgewichte anwenden und Gesamtverlust berechnen
    loss_c = args.content_weight * loss_c
    loss = loss_c + loss_gt + (l_identity1 * args.gt_weight) + (l_identity2 * args.gt_weight)

    # Vergleichsbilder speichern (alle 100 Iterationen)
    if i % 100 == 0:
        output_name = os.path.join(output_dir, f"{i}.jpg")
        comparison_image = torch.cat((content_images, out, gt_images), 0)  # Vergleich von Content, Output und Ground Truth
        save_image(comparison_image, output_name)

    # Verlust in der Deque speichern
    last_50_losses.append(loss.sum().item())

    # Gemittelten Verlust der letzten 50 Iterationen berechnen und ausgeben
    if (i + 1) % 50 == 0:
        avg_loss = sum(last_50_losses) / len(last_50_losses)
        print(f"Average loss over last 50 iterations: {avg_loss}")

    # Ausgabe der aktuellen Verluste
    print(loss.sum().cpu().detach().numpy(), "-content:", loss_c.sum().cpu().detach().numpy(),
          "-gt:", loss_gt.sum().cpu().detach().numpy(), "-l1:", l_identity1.sum().cpu().detach().numpy(),
          "-l2:", l_identity2.sum().cpu().detach().numpy())

    # Rückwärtsdurchführung und Optimierungsschritt
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    # TensorBoard-Logging der Verluste
    writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    writer.add_scalar('loss_gt', loss_gt.sum().item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    # Speichern des Modells in regelmäßigen Abständen
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.module.transformer.state_dict() if use_data_parallel else network.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, f'{args.save_dir}/transformer_iter_{i + 1}.pth')

        state_dict = network.module.decode.state_dict() if use_data_parallel else network.decode.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, f'{args.save_dir}/decoder_iter_{i + 1}.pth')

        state_dict = network.module.embedding.state_dict() if use_data_parallel else network.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, f'{args.save_dir}/embedding_iter_{i + 1}.pth')
