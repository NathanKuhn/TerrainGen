import numpy as np

WORLD_SIZE = (16, 16, 4)
CHUNK_SIZE = 32


def interp(array, x, y):
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1

    if x1 >= array.shape[0]:
        x1 = x0
    if y1 >= array.shape[1]:
        y1 = y0

    x -= x0
    y -= y0

    return (
        array[x0, y0] * (1 - x) * (1 - y)
        + array[x1, y0] * x * (1 - y)
        + array[x0, y1] * (1 - x) * y
        + array[x1, y1] * x * y
    )


def write_heightmap(heightmap, filename):
    with open(filename, "wb") as f:
        chunk_data = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint16)

        for z_chunk in range(WORLD_SIZE[2]):
            for y_chunk in range(WORLD_SIZE[1]):
                for x_chunk in range(WORLD_SIZE[0]):

                    for x in range(CHUNK_SIZE):
                        wx = x + x_chunk * CHUNK_SIZE
                        for y in range(CHUNK_SIZE):
                            wy = y + y_chunk * CHUNK_SIZE
                            height = interp(heightmap, wx / 2, wy / 2) / 15

                            for z in range(CHUNK_SIZE):
                                wz = z + z_chunk * CHUNK_SIZE
                                chunk_data[x, y, z] = 1 if wz < height else 0
                    
                    for z in range(CHUNK_SIZE):
                        for y in range(CHUNK_SIZE):
                            for x in range(CHUNK_SIZE):
                                if chunk_data[x, y, z] > 0:
                                    f.write(b"\x01\x00")
                                else:
                                    f.write(b"\x00\x00")


def main():
    heightmaps = np.load("dataset/terrain_dataset.npy")

    write_heightmap(
        heightmaps[1000],
        "save",
    )


if __name__ == "__main__":
    main()
