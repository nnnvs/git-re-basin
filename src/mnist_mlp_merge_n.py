from models.mlp import MLP
from utils.weight_matching import mlp_permutation_spec, weight_matching, apply_permutation
from utils.utils import flatten_params, lerp
from utils.plot import plot_interp_acc
import argparse
import random
import torch
from torchvision import datasets, transforms
from utils.training import test
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

from torch import randint

rngmix = lambda rng, x: randint(rng, x, (1,))
# returns prng key, and a function that takes a prng key and returns a new prng key
rngmix = lambda rng, x: random.fold_in(rng, hash(x))

def main():

  batch_size = 500
  learning_rate = 1e-3
  num_epochs = 100

  parser = argparse.ArgumentParser()
  parser.add_argument('--num_models', type=int, default=2)
  args = parser.parse_args()

  # Get data
  args = parser.parse_args()
  use_cuda = torch.cuda.is_available()

  seeds = random.randint(1, 500, size=args.num_models)
  device = torch.device("cuda" if use_cuda else "cpu")
  print("Device avaliable: ", device)

  dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
  dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform) 

  def train_one(seed: int):
    torch.manual_seed(seed)

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    torch.save(model.state_dict(), f"models/mnist_mlp_weights_{seed}.pt")

    return model

  models = [train_one(i) for i in seeds]

  # merge models
  permutation_spec = mlp_permutation_spec(4)

  def match_many(rng, permutation_spec: PermutationSpec, ps, max_iter=100):
    for iteration in range(max_iter):
      progress = False
      for p_ix in random.permutation(rngmix(rng, iteration), len(ps)):
        other_models_mean = tree_mean(ps[:p_ix] + ps[p_ix + 1:])
        l2_before = tree_l2(other_models_mean, ps[p_ix])
        perm = weight_matching(rngmix(rng, f"{iteration}-{p_ix}"),
                               permutation_spec,
                               flatten_params(other_models_mean),
                               flatten_params(ps[p_ix]),
                               silent=True)
        ps[p_ix] = unflatten_params(
            apply_permutation(permutation_spec, perm, flatten_params(ps[p_ix])))
        l2_after = tree_l2(other_models_mean, ps[p_ix])
        progress = progress or l2_after < l2_before - 1e-12
        print(f"iteration {iteration}/model {p_ix}: l2 diff {l2_after - l2_before:.4f}")

      if not progress:
        break

    return ps





























    ######################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # load models
    model_a = MLP()
    model_b = MLP()
    checkpoint = torch.load(args.model_a)
    model_a.load_state_dict(checkpoint)
    checkpoint_b = torch.load(args.model_b)
    model_b.load_state_dict(checkpoint_b)

    permutation_spec = mlp_permutation_spec(4)
    final_permutation = weight_matching(permutation_spec,
                                        flatten_params(model_a), flatten_params(model_b))
              

    updated_params = apply_permutation(permutation_spec, final_permutation, flatten_params(model_b))

    
    # test against mnist
    transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
    test_kwargs = {'batch_size': 5000}
    train_kwargs = {'batch_size': 5000}
    dataset = datasets.MNIST('../data', train=False,
                      transform=transform)
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                      transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)                  
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
    lambdas = torch.linspace(0, 1, steps=25)

    test_acc_interp_clever = []
    test_acc_interp_naive = []
    train_acc_interp_clever = []
    train_acc_interp_naive = []
    # naive
    model_b.load_state_dict(checkpoint_b)
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for lam in tqdm(lambdas):
      naive_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      test_loss, acc = test(model_b.cuda(), 'cuda', test_loader)
      test_acc_interp_naive.append(acc)
      train_loss, acc = test(model_b.cuda(), 'cuda', train_loader)
      train_acc_interp_naive.append(acc)

    # smart
    model_b.load_state_dict(updated_params)
    model_b.cuda()
    model_a.cuda()
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for lam in tqdm(lambdas):
      naive_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      test_loss, acc = test(model_b.cuda(), 'cuda', test_loader)
      test_acc_interp_clever.append(acc)
      train_loss, acc = test(model_b.cuda(), 'cuda', train_loader)
      train_acc_interp_clever.append(acc)

    fig = plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
                    train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig(f"mnist_mlp_weight_matching_interp_accuracy_epoch.png", dpi=300)

if __name__ == "__main__":
  main()