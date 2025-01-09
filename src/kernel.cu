// SHA 256 implementācija CUDA vidē
// https://en.wikipedia.org/wiki/SHA-2

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/std/cstdint> // analogs C/C++ <cstdint>, bet nodrošina fiksētus datu tipu lielumus uz device
#include <string>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <array>
#include "sha256_cpu.h"

// forward deklarācija funkcijām, lai nav intelisense warningi, ka tās nav definētas (ir pieejamas uz device bez header include)
__device__ unsigned int __funnelshift_r(unsigned int lo,unsigned int hi,unsigned int shift);
unsigned int atomicCAS(unsigned int* address,unsigned int compare,unsigned int val);

// macro priekš katra cuda API izsaukuma rezultāta pārbaudes
// ņemts no https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
	  fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  if (abort) exit(code);
   }
}

// ROTR(x,n) rotē x-a bitus pa labi pa n pozīcijām, izmantojam cuda iebūvēto funnelshift funkciju:
// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html 
// __device__ unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)
// Concatenate hi : lo , shift right by shift & 31 bits, return the least significant 32 bits. 
// Ja konkatenē x ar pašu sevi, tad pēc nobīdes, mazākie 32 biti saturēs attiecīgo ROTR no x
#define ROTR(x, n) __funnelshift_r(x, x, n)

// makro funkcijas attiecīgā sha bloka apstrādei
#define SS0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
#define SS1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))
#define S0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define CH(x, y, z)  ((x & y) ^ (~x & z))
#define MAJ(x, y, z) ((x & y) ^ (x & z) ^ (y & z))

// pirmie 32 biti kv. saknei no pirmajiem 8 pirmskaitļiem 2 - 19 (no daļas aiz komata)
__device__ cuda::std::uint32_t h0 = 0x6a09e667;
__device__ cuda::std::uint32_t h1 = 0xbb67ae85;
__device__ cuda::std::uint32_t h2 = 0x3c6ef372;
__device__ cuda::std::uint32_t h3 = 0xa54ff53a;
__device__ cuda::std::uint32_t h4 = 0x510e527f;
__device__ cuda::std::uint32_t h5 = 0x9b05688c;
__device__ cuda::std::uint32_t h6 = 0x1f83d9ab;
__device__ cuda::std::uint32_t h7 = 0x5be0cd19;

// pirmie 32 biti no kubsaknēm pirmajiem 64 pirmskaitļiem 2 - 311
__device__ __constant__ cuda::std::uint32_t k[] =
{
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// apstrādā vienu, konkrētu 512 bitu bloku 
// 'state' ir 8 skaitļu masīvs, kuram apstrādes sākumā jāsatur h0-h7 konstantes, apstrādes beigās saturēs hash vērtību
// 'chunk' satur apstrādājamo bitu bloku
__device__ void sha256ProcessChunk(cuda::std::uint32_t *state, cuda::std::uint8_t *chunk)
{
	cuda::std::uint32_t w[64];

	// iekopē visus 512 bitus iekš w masīva (512/32 = 16 vērtības)
	// baiti jāieliek iekš 32 bitu vārdiem, lai pirmais baits būtu pirmais (skatoties no kreisās uz labo pusi),
	// tas jāpabīda pa kreisi pa 24, nākamie pa 16, 8, 0
	// attiecīgā solī nākamie 'mazāksvarīgie' biti ir nulles, tāpēc baitus šos baitus var konkatenēt ar OR (|) operatoru 
	for(int i = 0; i < 16; i++)
	{
		w[i] = chunk[i * 4 + 0] << 24;
		w[i] |= chunk[i * 4 + 1] << 16;
		w[i] |= chunk[i * 4 + 2] << 8;
		w[i] |= chunk[i * 4 + 3];
	}

	// aizpilda pārējas 'w' vērtības
	for(int i = 16; i < 64; i++)
	{
		w[i] = w[i-16] + SS0(w[i-15]) + w[i-7] + SS1(w[i-2]);
	}

	cuda::std::uint32_t a = state[0];
	cuda::std::uint32_t b = state[1];
	cuda::std::uint32_t c = state[2];
	cuda::std::uint32_t d = state[3];
	cuda::std::uint32_t e = state[4];
	cuda::std::uint32_t f = state[5];
	cuda::std::uint32_t g = state[6];
	cuda::std::uint32_t h = state[7];

	for(int i = 0; i < 64; i++)
	{
		cuda::std::uint32_t temp1 = h + S1(e) + CH(e,f,g) + k[i] + w[i];
		cuda::std::uint32_t temp2 = S0(a) + MAJ(a,b,c);
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

__device__ void sha256(const cuda::std::uint8_t *input, cuda::std::uint64_t length, cuda::std::uint8_t *output)
{
	// vienkāršības pēc apstrādāsim viena bloka ietvaros, tāpēc, ņemot vērā ziņojuma garumu un padding,
	// ziņojuma garums nedrīkst būt lielāks par 440 bitiem, lai viss ietilpstu vienā 512 bitu blokā
	// https://crypto.stackexchange.com/questions/54852/what-happens-if-a-sha-256-input-is-too-long-longer-than-512-bits
	bool lengthOk = length <= (440/8);
	assert(lengthOk);

	cuda::std::uint32_t state[8] = {
		h0,h1,h2,h3,h4,h5,h6,h7
	};

	cuda::std::uint8_t chunk[64];

	// sākumā nonullējam bloku
	for(int i = 0; i < 64; i++) {
		chunk[i] = 0;
	}
	
	// ierakstām pašu ziņojumu
	for(int i = 0; i < length; i++) {
		chunk[i] = input[i];
	}
	
	// pēc prasībām ir jāpieliek '1' bits, pārējās baita vērtības attiecīgi ir nulles, atbilstoši SHA mainīgā 'K' prasībām
	chunk[length] = 0b10000000;

	// padding galā jāpieliek ziņojuma garums kā 64 bitu big-endian skaitlis
	for(int i = 1; i <= 8; i++)
	{
		chunk[64-i] = ((length * 8) >> ((i-1) * 8)) & 0xFF;
	}

	sha256ProcessChunk(state, chunk);

	// sadalām 32 bitu vērtības 4ās 8 bitu un ierakstām output masīvā
	for(int i = 0; i < 8; i++) {
		cuda::std::uint32_t currentStateValue = state[i];

		output[i * 4] = (cuda::std::uint8_t)(currentStateValue >> 24);
		output[i * 4 + 1] = (cuda::std::uint8_t)(currentStateValue >> 16);
		output[i * 4 + 2] = (cuda::std::uint8_t)(currentStateValue >> 8);
		output[i * 4 + 3] = (cuda::std::uint8_t)(currentStateValue);
	}
}

__device__ bool compareHashes(const cuda::std::uint8_t *h1, const cuda::std::uint8_t *h2)
{
	for(int i = 0; i < 32; i++)
	{
		if(h1[i] != h2[i])
		{
			return false;
		}
	}

	return true;
}

__global__ void kernel(
	const cuda::std::uint8_t *passwords,
	const int *pwLengths,
	int pwCount,
	cuda::std::uint64_t maxPwLength,
	const cuda::std::uint8_t *targetHash,
	int *resultIndex
)
{
	int idx = blockIdx.x * blockDim.x +threadIdx.x;

	if(idx >= pwCount)
	{
		return;
	}

	const cuda::std::uint8_t *password = passwords + (idx * maxPwLength);
	int pwLength = pwLengths[idx];

	cuda::std::uint8_t hash[32];


	sha256(password, pwLength, hash);

	// ja hashi sakrīt, tad ierakstām paroles indeksu iekš 'resultIndex',
	// tā kā tā mainīgā adrese atrodas device kopīgajā atmiņā, jālieto atomiska funkcija
	if(compareHashes(targetHash, hash))
	{
		// ja resultIndex glabājas vērtība -1, tad aizstāj to ar idx
		// ar šo tiek arī reizē nodrošināts, ka ja kāds pavediens vēlāk tomēr nonāk līdz šim stāvoklim,
		// tad modifikācija netiks veikta, jo tajā brīdī jau 'resultIndex' != -1
		atomicCAS((unsigned int*)resultIndex, -1, (unsigned int)idx);
	}
}

static uint8_t parseHexByte(const std::string &hash, size_t offset)
{
	std::string byteString = hash.substr(offset, 2);
	return static_cast<uint8_t>(std::stoi(byteString, nullptr, 16));
}

std::vector<uint8_t> hexStringToBytes(const std::string &hash)
{
	// 256 biti => 64 hex skaitļi
	if(hash.size() != 64)
	{
		throw std::runtime_error("SHA-256 hash as a hex string must be exactly 64 characters!");
	}
	
	std::vector<uint8_t> result(32);

	for(size_t i = 0; i < 32; i++)
	{
		result[i] =  parseHexByte(hash, i * 2);
	}

	return result;
}


std::string parseBytesToHexString(const uint8_t* data, size_t length)
{
	std::ostringstream ss;
	
	for(size_t i = 0; i < length; i++)
	{
		ss << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
	}

	return ss.str();
}

void hashCheck(std::vector<std::string>& passwords, std::vector<uint8_t>& hash, int *cracked_idx, bool useGpu)
{
	assert(*cracked_idx == -1); // te sākumā jau jābūt vērtībai -1, padota no main

	const int batchSize = 1 << 18; // 1D režģiem cuda limitācija ir 2^31, bet šeit limitējošais faktors būs atmiņa parolēm

	const int passwordCount = passwords.size();
	const int maxPwLength = 55; // maksimālais ziņojuma garums, lai tas ietilptu vienā sha blokā
	const size_t maxBatchBufferSize = batchSize * maxPwLength;

	std::vector<uint8_t> pwBuffer(maxBatchBufferSize, 0);
	std::vector<int> pwLengths(batchSize);


	if(useGpu)
	{
		cuda::std::uint8_t *d_passwords;
		cuda::std::uint8_t *d_hash;
		int *d_pwLengths;
		int *d_cracked_idx;

		*cracked_idx = -1;
		
		CUDA_CHECK(cudaSetDevice(0));

		CUDA_CHECK(cudaMalloc(&d_hash, 32));
		CUDA_CHECK(cudaMemcpy(d_hash,hash.data(), 32, cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMalloc(&d_cracked_idx, sizeof(int)));
		CUDA_CHECK(cudaMemcpy(d_cracked_idx, cracked_idx, sizeof(int), cudaMemcpyHostToDevice));

		for(int batchStart = 0; batchStart < passwordCount; batchStart += batchSize)
		{
			// pēdējais batch var neaizņemt pilnu apjomu
			int currentBatchSize = std::min(batchSize,passwordCount - batchStart); 

			// batcham atbilstošo paroļu ievietošana buferī
			for(size_t i = 0; i < currentBatchSize; i++) {
				const std::string& pw = passwords[batchStart + i];
				int pwLength = pw.size();

				assert(pwLength <= maxPwLength);

				pwLengths[i] = pwLength;
				
				for(size_t j = 0; j < pwLength; j++)
				{
					// ja parole ir īsāka par maxPwLength, tad 'tukšie' masīva elementi būs aizpildīti ar nullēm
					pwBuffer[i * maxPwLength + j] = (uint8_t)pw[j]; 

				}
			}

			CUDA_CHECK(cudaMalloc(&d_passwords, currentBatchSize * maxPwLength * sizeof(cuda::std::uint8_t)));
			CUDA_CHECK(cudaMemcpy(d_passwords, pwBuffer.data(), currentBatchSize * maxPwLength * sizeof(cuda::std::uint8_t), cudaMemcpyHostToDevice));

			CUDA_CHECK(cudaMalloc(&d_pwLengths, currentBatchSize * sizeof(int)));
			CUDA_CHECK(cudaMemcpy(d_pwLengths, pwLengths.data(), currentBatchSize * sizeof(int), cudaMemcpyHostToDevice));

			int numThreads = 256;
			int numBlocks = (currentBatchSize + numThreads - 1 ) / numThreads;

			kernel<<<numBlocks, numThreads>>>(d_passwords, d_pwLengths, passwordCount, maxPwLength, d_hash, d_cracked_idx);

			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());

			CUDA_CHECK(cudaMemcpy(cracked_idx, d_cracked_idx, sizeof(int), cudaMemcpyDeviceToHost));

			cudaFree(d_passwords);
			cudaFree(d_pwLengths);

			if(*cracked_idx != -1)
			{
				*cracked_idx += batchStart; // indekss ir relatīvs batcham, tāpēc vajag offsetu pieskaitīt
				goto cuda_free_and_exit; // dirty risinājums, bet strādā
			}
		}

		cuda_free_and_exit: 
		cudaFree(d_hash);
		cudaFree(d_cracked_idx);
	}
	// CPU izpilde
	else
	{
		// līdzīga batch-veidīga apstrāde kā gpu
		for (int batchStart = 0; batchStart < passwordCount; batchStart += batchSize) {
            int currentBatchSize = std::min(batchSize, passwordCount - batchStart);

            for (int i = 0; i < currentBatchSize; i++) {
                const std::string& pw = passwords[batchStart + i];
                size_t pwLength = pw.size();

                assert(pwLength <= maxPwLength);
                pwLengths[i] = pwLength;

                for (size_t j = 0; j < pwLength; j++) {
                    pwBuffer[i * maxPwLength + j] = (uint8_t)pw[j];
                }
            }

            for (int i = 0; i < currentBatchSize; i++) {
                const uint8_t* pwStart = &pwBuffer[i * maxPwLength];
                size_t pwLength = pwLengths[i];

                std::vector<uint8_t> computedHash(32);

                cpu_sha256(pwStart, pwLength, computedHash.data());

                if (computedHash == hash) {
                    *cracked_idx = batchStart + i;
                    return;
                }
            }
        }
        *cracked_idx = -1;
	}

}


void processFile(const std::string& fileName, std::vector<std::string>& buffer)
{
	std::ifstream file(fileName);

	if(!file.is_open())
	{
		throw std::runtime_error("Could not open file " + fileName);
	}

	std::string line;
	unsigned int currentOffset = 0;
	
	while(std::getline(file,line))
	{
		buffer.push_back(line);
	}

	file.close();
}

// sha funkcijas testa device kodols
__global__ void testKernel(
	const cuda::std::uint8_t *input,
	cuda::std::uint64_t length,
	cuda::std::uint8_t *calculatedHash
)
{
	// testam pietieks ar pirmo pavedienu
	if(threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}

	sha256(input,length,calculatedHash);
}

void testSha(const std::string &password,const std::string hexExpectedHash)
{
	std::vector<uint8_t> expectedHash = hexStringToBytes(hexExpectedHash);

	size_t pwLength = password.size();
	std::vector<uint8_t> passwordBytes(pwLength);

	for(size_t i = 0; i < pwLength; i++)
	{
		passwordBytes[i] = password[i];
	}

	cuda::std::uint8_t *d_password;
	cuda::std::uint8_t *d_hash;
	cuda::std::uint8_t *d_calculatedHash;

	CUDA_CHECK(cudaSetDevice(0));

	CUDA_CHECK(cudaMalloc(&d_password,pwLength));
	CUDA_CHECK(cudaMemcpy(d_password, passwordBytes.data(),pwLength,cudaMemcpyHostToDevice));
	
	CUDA_CHECK(cudaMalloc(&d_calculatedHash,32));
	CUDA_CHECK(cudaMemcpy(d_calculatedHash, std::vector<uint8_t>(32,0).data(), 32, cudaMemcpyHostToDevice));


	int numThreads = 1;
	int numBlocks = 1;

	testKernel<<<numBlocks, numThreads>>>(d_password,pwLength,d_calculatedHash);
	
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	cuda::std::uint8_t h_calculatedHash[32];

	CUDA_CHECK(cudaMemcpy(&h_calculatedHash, d_calculatedHash, 32, cudaMemcpyDeviceToHost));

	cudaFree(d_password);
	cudaFree(d_calculatedHash);

	std::cout << "Expected:\t" << hexExpectedHash << "\nActual:\t\t" << parseBytesToHexString(h_calculatedHash,32) << '\n';
}


int main(int argc, char* argv[])
{
	try {
		// testu palaišana
		if(argc == 2 && std::string(argv[1]) == "--test")
		{
			std::cout << "SHA Tests\n";
		
			testSha("","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
			testSha("123456","8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92");


			std::cout << "Hash Converison Tests\n";

			std::string testHexHash = "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92";
			auto hexbytes = hexStringToBytes(testHexHash);
			std::cout << "Original hash:\t\t" << testHexHash << "\nRoundtrip converted: \t" << parseBytesToHexString(hexbytes.data(),hexbytes.size()) << '\n';

			std::cout << "Tests complete\n";
		}
		// padoti 4 argumenti, CPU vai GPU izpilde un paroles fails, hash
		else if(argc == 4)
		{
			const std::string cpuOrGpu = argv[1];
			const std::string inputFileName = argv[2];
			const std::string hexHash = argv[3];

			std::vector<std::string> buffer;
			processFile(inputFileName, buffer);

			std::vector<uint8_t> hash = hexStringToBytes(hexHash);

			std::cout << "Starting search...\n";

			int cracked_idx = -1;

			if (cpuOrGpu == "--cpu") {
				std::cout << "CPU mode\n";
				hashCheck(buffer, hash, &cracked_idx, false);
			} else if (cpuOrGpu == "--gpu") {
				std::cout << "GPU mode\n";
				hashCheck(buffer, hash, &cracked_idx, true);
			} else {
				throw std::runtime_error("Invalid mode! Use --cpu or --gpu");
			}


			if (cracked_idx != -1)
			{
				std::cout << "Password found!\n"
					<< buffer[cracked_idx] << std::endl;
			}
			else
			{
				std::cout << "No matching password found." << std::endl;
			}

		}
		else
		{
			std::cout << "Correct program usage:\n"
				<< "\tRunning tests:\n"
				<< "\t\t" << argv[0] << " --test\n"
				<< "\tCPU Password cracking:\n"
				<< "\t\t" << argv[0] << " --cpu <passwords file> <password hash>\n"
				<< "\tGPU Password cracking:\n"
				<< "\t\t" << argv[0] << " --gpu <passwords file> <password hash>\n";

			return -1;
		}
	}
	catch (const std::runtime_error &e)
	{
		std::cerr << "Error! " << e.what() << std::endl;
		return 1;
	}

	CUDA_CHECK(cudaDeviceReset());

	return 0; 
}

