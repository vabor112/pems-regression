import torch

def dcn(x):
    return x.detach().cpu().numpy()


def extract_node_coordinates(G, save_path="node_coordinates.pt"):
    """
    Extract node coordinates from an OSMnx graph into an (N, 2) torch tensor,
    ordered by ascending node ID.

    Args:
        G: OSMnx graph object with node attributes "x" (longitude) and "y" (latitude)
        save_path: path to save the tensor (default: "node_coordinates.pt")

    Returns:
        coords: (N, 2) float tensor, columns are [longitude, latitude],
                rows ordered by ascending node ID
    """
    sorted_nodes = sorted(G.nodes)

    coords = torch.tensor(
        [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in sorted_nodes],
        dtype=torch.float64,
    )  # shape (N, 2): [:, 0] = longitude, [:, 1] = latitude

    torch.save(coords, save_path)
    print(f"Saved coordinate tensor of shape {coords.shape} to '{save_path}'")

