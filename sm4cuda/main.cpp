#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sm4cuda.cuh"

using namespace std;

#define POS	 109

#define LEN  48

int main()
{
	//��Կ
	uint8_t key[16] = { 0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10 };

	//����
	uint8_t input_sample[16] = { 0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10 };

	//��֤����
	uint8_t add[16] = { 0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10 };

	//iv
	uint8_t iv_sample[16] = { 0xAA,0xAA,0xAA,0xAA,0xAA,0xAB,0xAA,0xAA,0xEF,0xAA,0xAA,0xAA,0x00,0x00,0x00,0x00 };

	device_memory *way = new device_memory();

	//���ü�����Կ
	sm4_setkey_enc(&way->ctx, key);

	//GPUģʽ
	uint8_t *input = (uint8_t *)malloc(PARTICLE_SIZE);
	uint8_t *output = (uint8_t *)malloc(PARTICLE_SIZE);
	uint8_t *tag = (uint8_t *)malloc(PARTICLE_SIZE);


	for (int i = 0; i < PARTICLE_SIZE / 16; i++)
	{
		memcpy(input + 16 * i, input_sample, 16);
	}

	//��ʼ���豸�ڴ�
	Init_device_memory(way, add, iv_sample);

	
	///////////////////////////////////////////////////////////////////
	//�ٶȲ��Կ�ʼ
	///////////////////////////////////////////////////////////////////
	clock_t  clockBegin, clockEnd;
	clockBegin = clock();
	for (int i = 0; i < LEN; i++)
	{
		sm4_gcm_enc(way, i + 1, input, output);
	}
	clockEnd = clock();
	printf("GPU use %d ms\n", clockEnd - clockBegin);
	double calctime = (clockEnd - clockBegin) / 1000.0;
	double speed = LEN * 0.03125 / calctime;
	printf("GPU speed: %.2f GB/s\n", speed);
	///////////////////////////////////////////////////////////////////
	//�ٶȲ��Խ���
	///////////////////////////////////////////////////////////////////

	sm4_gcm_final(way, LEN * PARTICLE_SIZE, tag);

	Free_device_memory(way);

	return 0;
}

