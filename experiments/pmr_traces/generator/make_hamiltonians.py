"""Hamiltonian input files (PMR-QMC format: "J i P j P ...") for the traces."""
def heis2d(L, fn):
    idx = lambda x, y: (x % L) + (y % L) * L + 1
    bonds = set()
    for x in range(L):
        for y in range(L):
            for dx, dy in [(1, 0), (0, 1)]:
                a, b = idx(x, y), idx(x + dx, y + dy); bonds.add((min(a, b), max(a, b)))
    with open(fn, 'w') as f:
        for a, b in sorted(bonds):
            for P in 'XYZ': f.write(f"1.0 {a} {P} {b} {P}\n")
def tfim(N, h, fn):
    with open(fn, 'w') as f:
        for i in range(1, N + 1): f.write(f"-1.0 {i} Z {i % N + 1} Z\n")
        for i in range(1, N + 1): f.write(f"-{h} {i} X\n")
heis2d(4, 'H_heis4x4.txt'); heis2d(8, 'H_heis8x8.txt')
tfim(32, 1.0, 'H_tfim32.txt'); tfim(64, 1.0, 'H_tfim64.txt')
