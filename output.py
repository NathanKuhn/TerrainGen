import numpy as np

WORLD_SIZE = (16, 16, 4)
CHUNK_SIZE = 64

I16_MAX = 2**15 - 1


def sample_height(array: np.ndarray, x: float, y: float) -> float:
    """
    Sample the array at the given coordinates using linear interpolation.
    :param array: The array to sample W x H.
    :param x: The x coordinate (0, 1).
    :param y: The y coordinate (0, 1).
    :return: The value at the given coordinates.
    """
    assert 0 <= x <= 1, f"x: {x} not in [0, 1]"
    assert 0 <= y <= 1, f"y: {y} not in [0, 1]"

    x = x * (array.shape[0] - 1)
    y = y * (array.shape[1] - 1)

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


def sample_normal(
    array: np.ndarray, x: float, y: float, scale=1.0
) -> tuple[float, float, float]:
    """
    Get the normal at the given coordinates.
    :param array: The array to sample W x H.
    :param x: The x coordinate (0, 1).
    :param y: The y coordinate (0, 1).
    :return: The normal at the given coordinates.
    """

    x0 = sample_height(array, max(x - 0.01, 0), y)
    x1 = sample_height(array, min(x + 0.01, 1), y)
    y0 = sample_height(array, x, max(y - 0.01, 0))
    y1 = sample_height(array, x, min(y + 0.01, 1))

    dzdx = (x1 - x0) / 0.02
    dzdy = (y1 - y0) / 0.02

    mag = np.sqrt(dzdx**2 + dzdy**2 + scale)

    return -dzdx / mag, -dzdy / mag, 1 / mag


def write_heightmap(heightmap: np.ndarray, filename):
    min_height = np.min(heightmap)
    max_height = np.max(heightmap)

    heightmap = heightmap.astype(np.float32)
    heightmap -= min_height
    heightmap /= max_height - min_height

    with open(filename, "wb") as f:
        block_data = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint16)
        normal_data = np.zeros((CHUNK_SIZE, CHUNK_SIZE, 3), dtype=np.float32)

        for z_chunk in range(WORLD_SIZE[2]):
            for y_chunk in range(WORLD_SIZE[1]):
                for x_chunk in range(WORLD_SIZE[0]):

                    for x in range(CHUNK_SIZE):
                        wx = x + x_chunk * CHUNK_SIZE
                        for y in range(CHUNK_SIZE):
                            wy = y + y_chunk * CHUNK_SIZE

                            fx = wx / (CHUNK_SIZE * WORLD_SIZE[0])
                            fy = wy / (CHUNK_SIZE * WORLD_SIZE[1])

                            height = sample_height(heightmap, fx, fy) * CHUNK_SIZE

                            for z in range(CHUNK_SIZE):
                                wz = z + z_chunk * CHUNK_SIZE
                                block_data[x, y, z] = 1 if wz < height else 0

                            normal_data[x, y] = sample_normal(
                                heightmap, fx, fy, scale=CHUNK_SIZE
                            )

                    for z in range(CHUNK_SIZE):
                        for y in range(CHUNK_SIZE):
                            for x in range(CHUNK_SIZE):
                                if block_data[x, y, z] > 0:
                                    f.write(b"\x02\x00")
                                else:
                                    f.write(b"\x00\x00")

                    for y in range(CHUNK_SIZE):
                        for x in range(CHUNK_SIZE):
                            nx, ny, _ = normal_data[x, y]
                            nx = int(nx * I16_MAX)
                            ny = int(ny * I16_MAX)

                            f.write(nx.to_bytes(2, "little", signed=True))
                            f.write(ny.to_bytes(2, "little", signed=True))


def main():
    heightmaps = np.load("dataset/terrain_dataset.npy")

    write_heightmap(
        heightmaps[8],
        "D:\\Programming\\Rust\\circuit-game\\target\\debug\\save",
    )


if __name__ == "__main__":
    main()
