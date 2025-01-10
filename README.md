# Kursa darba "GPU Programmēšana NVIDIA CUDA un AMD ROCm Saskarnēs" raktiskā uzdevuma realizācija

**Autors:** Artūrs Kļaviņš


## Projekta Struktūra

- Projekts izstrādāts Visual Studio, tāpēc to iespējams atvērt kā VS projektu
- Noklusēti tiks kompilēts un palaists CUDA variants


## Kompilēšana

### Visual Studio 2022
- Uz Windows noklusēti iespējams kompilēt projektu, izmantojot Visual Studio projekta konfigurāciju.

### Manuāla kompilācija no `./src` direktorijas (priekš Linux)

#### CUDA
```
nvcc -O2 kernel.cu sha256_cpu.cpp -o pwcracker
```

#### ROCm
```
hipcc -O2 kernel.hip sha256_cpu.cpp -o pwcracker
```

## Python skripti

### Nejaušu paroļu faila ģenerēšana:
```
python3 ./src/pwgen.py <vēlamais_paroļu_skaits> <izvades_faila_vārds>
```

### `pwcracker` programmas benchmarkings:
```
python3 ./src/benchmark.py
```

## Pārbaudīts uz:
- Windows 11 (tikai CUDA variants!)
- Ubuntu 24.04 LTS
- RTX 3060
- CUDA 12.6
- ROCM 6.2.3
