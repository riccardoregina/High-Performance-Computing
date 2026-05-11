import math

def occupancy(d, l) -> float:
    """Calculate occupancy
       
       d: the device configuration,
       l: the launch configuration
    """
    # Input validation
    if any(size > max for size, max in zip(l['blockSize'], d['maxBlockSize'])):
        raise ValueError('Illegal launch blocksize')
    if any(size > max for size, max in zip(l['gridSize'], d['maxGridSize'])):
        raise ValueError('Illegal launch gridsize')
    # Calculate core values
    blocks = math.prod(l['gridSize'])
    print(f'Wanna run {blocks} blocks')
    threadsPerBlock = math.prod(l['blockSize'])
    warpsPerBlock = threadsPerBlock // 32
    # Limit: number of blocks
    blocks = min(blocks, d['blocksPerSM'])
    print(f'After checking max number of blocks: {blocks}')
    # Limit: registers
    blocks = min(
        blocks,
        d['registersPerSM'] // (threadsPerBlock * l['registersPerThread'])
    )
    print(f'After checking max number of registers: {blocks}')
    # Limit: shared memory
    blocks = min(blocks, d['sharedMemSizeByte'] // l['sharedMemPerBlock'])
    print(f'After checking max shared memory size: {blocks}')
    # Limit: number of threads
    blocks = min(blocks, d['threadsPerSM'] // threadsPerBlock)
    print(f'After checking max number of threads: {blocks}')
    # Limit: number of warps
    blocks = min(blocks, d['warpsPerSM'] // warpsPerBlock)
    print(f'After checking max number of warps: {blocks}')
    # Calculate occupancy
    warps = blocks * warpsPerBlock
    print(f"Active warps: {warps}/{d['warpsPerSM']}")
    occupancy = warps / d['warpsPerSM']
    return occupancy


gtx1060 = {
        'SMs': 10,
        'threadsPerSM': 2048,
        'warpsPerSM': 64,
        'blocksPerSM': 32,
        'registersPerSM': 65536,
        'registerSizeByte': 4,
        'sharedMemSizeByte': 49152,
        'globalMemSizeByte': 6000000000,
        'maxBlockSize': (1024, 1024, 64),
        'maxGridSize': (2147483647, 65535, 65535),
}

launch = {
        'blockSize': (64, 1, 1),
        'gridSize': (16, 1, 1),
        'registersPerThread': 27,
        'sharedMemPerBlock': 1024,
}

print(f'Occupancy: {occupancy(gtx1060, launch)}')
