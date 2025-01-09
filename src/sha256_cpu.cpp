// CPU puses implementācija SHA256, skaidrojumus skatīt kernel.cu failā

#include "sha256_cpu.h"
#include <cstring>
#include <cassert>
#include <cstdio>
#include <array>
#include <vector>

constexpr uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

uint32_t ROTR(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

uint32_t CH(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

uint32_t MAJ(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

uint32_t S0(uint32_t x) {
    return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
}

uint32_t S1(uint32_t x) {
    return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
}

uint32_t SS0(uint32_t x) {
    return ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3);
}

uint32_t SS1(uint32_t x) {
    return ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10);
}

void cpu_sha256ProcessChunk(uint32_t *state, const uint8_t *chunk) {
    uint32_t w[64];

    for (int i = 0; i < 16; ++i) {
        w[i] = (chunk[i * 4] << 24) |
               (chunk[i * 4 + 1] << 16) |
               (chunk[i * 4 + 2] << 8) |
               (chunk[i * 4 + 3]);
    }

    for (int i = 16; i < 64; ++i) {
        w[i] = w[i - 16] + SS0(w[i - 15]) + w[i - 7] + SS1(w[i - 2]);
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t temp1 = h + S1(e) + CH(e, f, g) + k[i] + w[i];
        uint32_t temp2 = S0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void cpu_sha256(const uint8_t *input, size_t length, uint8_t *output) {
    assert(length <= (440 / 8));

    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint8_t chunk[64] = {0};

    memcpy(chunk, input, length);
    chunk[length] = 0x80;

    uint64_t bitLength = length * 8;
    for (int i = 0; i < 8; ++i) {
        chunk[63 - i] = bitLength >> (i * 8);
    }

    cpu_sha256ProcessChunk(state, chunk);

    for (int i = 0; i < 8; ++i) {
        output[i * 4] = (state[i] >> 24) & 0xFF;
        output[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        output[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        output[i * 4 + 3] = state[i] & 0xFF;
    }
}
