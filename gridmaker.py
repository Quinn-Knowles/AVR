import matplotlib.pyplot as plt

def generate_grid_image(grid_size=10, cell_size=1, filename='grid.png'):
    fig, ax = plt.subplots(figsize=(grid_size, grid_size))

    for x in range(grid_size + 1):
        ax.axvline(x * cell_size, color='black', linewidth=1)

    for y in range(grid_size + 1):
        ax.axhline(y * cell_size, color='black', linewidth=1)

    # Set limits and aspect ratio
    ax.set_xlim(0, grid_size * cell_size)
    ax.set_ylim(0, grid_size * cell_size)
    ax.set_aspect('equal')
    ax.axis('off') 

    # Save the image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Grid image saved as {filename}")

generate_grid_image(grid_size=50, cell_size=1, filename='square_grid.png')