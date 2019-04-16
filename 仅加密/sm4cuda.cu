#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sm4cuda.cuh"

//S�в���
uint8_t SboxTable[256] = { \
	0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05, \
	0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99, \
	0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62, \
	0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6, \
	0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8, \
	0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35, \
	0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87, \
	0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e, \
	0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1, \
	0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3, \
	0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f, \
	0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51, \
	0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8, \
	0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0, \
	0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84, \
	0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48, \
};

/* System parameter */
uint32_t FK[4] = { 0xa3b1bac6,0x56aa3350,0x677d9197,0xb27022dc };

/* fixed parameter */
uint32_t CK[32] = { \
	0x00070e15,0x1c232a31,0x383f464d,0x545b6269, \
	0x70777e85,0x8c939aa1,0xa8afb6bd,0xc4cbd2d9, \
	0xe0e7eef5,0xfc030a11,0x181f262d,0x343b4249, \
	0x50575e65,0x6c737a81,0x888f969d,0xa4abb2b9, \
	0xc0c7ced5,0xdce3eaf1,0xf8ff060d,0x141b2229, \
	0x30373e45,0x4c535a61,0x686f767d,0x848b9299, \
	0xa0a7aeb5,0xbcc3cad1,0xd8dfe6ed,0xf4fb0209, \
	0x10171e25,0x2c333a41,0x484f565d,0x646b7279, \
};

/*
   ����λ���� C++�汾
   b:��Ҫ�ƶ�������ָ��
   i:��Ҫ�ƶ���λ��
   n:����ֵ��
 */
inline void GET_UINT_BE(uint32_t *n, uint8_t *b, uint32_t i)
{
	(*n) = (((uint32_t)b[i]) << 24) | (((uint32_t)b[i + 1]) << 16) | (((uint32_t)b[i + 2]) << 8) | (uint32_t)b[i + 3];
}

/*
	����λ���� C++�汾������
	b:��Ҫ�ƶ�������ָ��
	i:��Ҫ�ƶ���λ��
	n:����ֵ��
*/
inline void PUT_UINT_BE(uint32_t n, uint8_t *b, uint32_t i)
{
	//ȡn�ĸ���λ
	b[i + 0] = (uint8_t)(n >> 24);

	//ȡn�Ĵθ���λ
	b[i + 1] = (uint8_t)(n >> 16);

	//ȡn�Ĵε���λ
	b[i + 2] = (uint8_t)(n >> 8);

	//ȡn�ĵ���λ
	b[i + 3] = (uint8_t)n;
}

/*
	S���滻
*/
inline uint8_t sm4Sbox(uint8_t inch)
{
	return SboxTable[inch];
}

/*
	ѭ�����ƺ���������xѭ������nλ
*/
inline uint32_t ROTL(uint32_t x, uint32_t n)
{
	return (x << n) | (x >> (32 - n));
}

/*
	����a b��ֵ
*/
inline void SWAP(uint32_t *a, uint32_t *b)
{
	uint32_t c = *a;
	*a = *b;
	*b = c;
}

uint32_t sm4Lt(uint32_t ka)
{
	uint8_t a[4];
	PUT_UINT_BE(ka, a, 0);

	//����滻
	a[0] = sm4Sbox(a[0]);
	a[1] = sm4Sbox(a[1]);
	a[2] = sm4Sbox(a[2]);
	a[3] = sm4Sbox(a[3]);

	//����������ŵ�bb������ȥ
	uint32_t bb = 0;
	GET_UINT_BE(&bb, a, 0);

	//bb�ֱ�����ѭ������2λ��10λ��18λ��24λ������� �õ���ֵ����
	return bb ^ (ROTL(bb, 2)) ^ (ROTL(bb, 10)) ^ (ROTL(bb, 18)) ^ (ROTL(bb, 24));
}

uint32_t sm4F(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t rk)
{
	return (x0^sm4Lt(x1^x2^x3^rk));
}


/*
	��Կ��չ����
*/
uint32_t sm4CalciRK(uint32_t ka)
{
	uint8_t a[4];
	PUT_UINT_BE(ka, a, 0);
	a[0] = sm4Sbox(a[0]);
	a[1] = sm4Sbox(a[1]);
	a[2] = sm4Sbox(a[2]);
	a[3] = sm4Sbox(a[3]);

	uint32_t bb = 0;
	GET_UINT_BE(&bb, a, 0);
	return bb ^ (ROTL(bb, 13)) ^ (ROTL(bb, 23));
}

/*
	SK:ֵ���������������д��չ��Կ
	key:��ʼ��Կ(128bit)
*/
void sm4_setkey(uint32_t SK[32], uint8_t key[16])
{
	uint32_t MK[4];
	GET_UINT_BE(&MK[0], key, 0);
	GET_UINT_BE(&MK[1], key, 4);
	GET_UINT_BE(&MK[2], key, 8);
	GET_UINT_BE(&MK[3], key, 12);


	//��ʼ����Կ
	uint32_t k[36];
	k[0] = MK[0] ^ FK[0];
	k[1] = MK[1] ^ FK[1];
	k[2] = MK[2] ^ FK[2];
	k[3] = MK[3] ^ FK[3];

	for (int i = 0; i < 32; i++)
	{
		k[i + 4] = k[i] ^ (sm4CalciRK(k[i + 1] ^ k[i + 2] ^ k[i + 3] ^ CK[i]));
		SK[i] = k[i + 4];
	}
}

/*
	SM4�ֺ���
*/
void sm4_one_round(uint32_t sk[32], uint8_t input[16], uint8_t output[16])
{

	uint32_t ulbuf[36];
	memset(ulbuf, 0, sizeof(ulbuf));

	GET_UINT_BE(&ulbuf[0], input, 0);
	GET_UINT_BE(&ulbuf[1], input, 4);
	GET_UINT_BE(&ulbuf[2], input, 8);
	GET_UINT_BE(&ulbuf[3], input, 12);

	for (int i = 0; i < 32; i++)
	{
		ulbuf[i + 4] = sm4F(ulbuf[i], ulbuf[i + 1], ulbuf[i + 2], ulbuf[i + 3], sk[i]);
	}

	PUT_UINT_BE(ulbuf[35], output, 0);
	PUT_UINT_BE(ulbuf[34], output, 4);
	PUT_UINT_BE(ulbuf[33], output, 8);
	PUT_UINT_BE(ulbuf[32], output, 12);
}

/*
	����ģʽ��Կ��չ
	ctx��ֵ�������������ִ����Ϻ����д������Կ�����Ϣ��
	key: ������Կ������128bit��
*/
void sm4_setkey_enc(sm4_context *ctx, uint8_t key[16])
{
	ctx->mode = SM4_ENCRYPT;
	sm4_setkey(ctx->sk, key);
}

/*
	����ģʽ��Կ��չ
	ctx��ֵ�������������ִ����Ϻ����д������Կ�����Ϣ��
	key: ������Կ������128bit��
*/
void sm4_setkey_dec(sm4_context *ctx, uint8_t key[16])
{
	ctx->mode = SM4_DECRYPT;
	sm4_setkey(ctx->sk, key);
	for (int i = 0; i < 16; i++)
	{
		SWAP(&(ctx->sk[i]), &(ctx->sk[31 - i]));
	}
}

/*
 * SM4-ECB block encryption/decryption
 *
 * SM4-ECBģʽ�ӽ��ܺ���
 * ctx��ֵ�������������Կ����ָ��
 * mode:�ӽ���ģʽ��SM4�����ּӽ���ģʽ�����Ľ������ĳ������Ľ������ĳ�
 * input:��������(16�ֽ�)
 * output:�������(16�ֽ�)
 */
void sm4_crypt_ecb(sm4_context *ctx, int length, uint8_t *input, uint8_t *output)
{
	while (length > 0)
	{
		sm4_one_round(ctx->sk, input, output);
		input += 16;
		output += 16;
		length -= 16;
	}
}

/*
	Ϊ����洢��ô��ͻʱ����Ҫ�����ݿ���д洢ģʽת����
	ת����Ҫ�Ĵ��ڴ�����������󣬹���Ҫ�ĸ����ұ�
	����������ת�����õĲ��ұ�
*/

//ת�����ұ�0
uint32_t matrix_table_zero[32] = {
	 0 * 4 + 0 * 128,  1 * 4 + 0 * 128,  2 * 4 + 0 * 128,  3 * 4 + 0 * 128, \
	 4 * 4 + 0 * 128,  5 * 4 + 0 * 128,  6 * 4 + 0 * 128,  7 * 4 + 0 * 128, \
	 8 * 4 + 1 * 128,  9 * 4 + 1 * 128, 10 * 4 + 1 * 128, 11 * 4 + 1 * 128, \
	12 * 4 + 1 * 128, 13 * 4 + 1 * 128, 14 * 4 + 1 * 128, 15 * 4 + 1 * 128, \
	16 * 4 + 2 * 128, 17 * 4 + 2 * 128, 18 * 4 + 2 * 128, 19 * 4 + 2 * 128, \
	20 * 4 + 2 * 128, 21 * 4 + 2 * 128, 22 * 4 + 2 * 128, 23 * 4 + 2 * 128, \
	24 * 4 + 3 * 128, 25 * 4 + 3 * 128, 26 * 4 + 3 * 128, 27 * 4 + 3 * 128, \
	28 * 4 + 3 * 128, 29 * 4 + 3 * 128, 30 * 4 + 3 * 128, 31 * 4 + 3 * 128, \
};
uint32_t linear_table_zero[32] = {
	 0 * 4 + 0 * 128,  4 * 4 + 0 * 128,  8 * 4 + 0 * 128, 12 * 4 + 0 * 128, \
	 16 * 4 + 0 * 128, 20 * 4 + 0 * 128, 24 * 4 + 0 * 128, 28 * 4 + 0 * 128,\
	 1 * 4 + 1 * 128,  5 * 4 + 1 * 128,  9 * 4 + 1 * 128, 13 * 4 + 1 * 128, \
	17 * 4 + 1 * 128, 21 * 4 + 1 * 128, 25 * 4 + 1 * 128, 29 * 4 + 1 * 128, \
	 2 * 4 + 2 * 128,  6 * 4 + 2 * 128, 10 * 4 + 2 * 128, 14 * 4 + 2 * 128, \
	18 * 4 + 2 * 128, 22 * 4 + 2 * 128, 26 * 4 + 2 * 128, 30 * 4 + 2 * 128, \
	 3 * 4 + 3 * 128,  7 * 4 + 3 * 128, 11 * 4 + 3 * 128, 15 * 4 + 3 * 128, \
	19 * 4 + 3 * 128, 23 * 4 + 3 * 128, 27 * 4 + 3 * 128, 31 * 4 + 3 * 128, \
};
//ת�����ұ�1
uint32_t matrix_table_one[32] = {
	 0 * 4 + 1 * 128,  1 * 4 + 1 * 128,  2 * 4 + 1 * 128,  3 * 4 + 1 * 128, \
	 4 * 4 + 1 * 128,  5 * 4 + 1 * 128,  6 * 4 + 1 * 128,  7 * 4 + 1 * 128, \
	 8 * 4 + 2 * 128,  9 * 4 + 2 * 128, 10 * 4 + 2 * 128, 11 * 4 + 2 * 128, \
	12 * 4 + 2 * 128, 13 * 4 + 2 * 128, 14 * 4 + 2 * 128, 15 * 4 + 2 * 128, \
	16 * 4 + 3 * 128, 17 * 4 + 3 * 128, 18 * 4 + 3 * 128, 19 * 4 + 3 * 128, \
	20 * 4 + 3 * 128, 21 * 4 + 3 * 128, 22 * 4 + 3 * 128, 23 * 4 + 3 * 128, \
	24 * 4 + 0 * 128, 25 * 4 + 0 * 128, 26 * 4 + 0 * 128, 27 * 4 + 0 * 128, \
	28 * 4 + 0 * 128, 29 * 4 + 0 * 128, 30 * 4 + 0 * 128, 31 * 4 + 0 * 128, \
};
uint32_t linear_table_one[32] = {
	 1 * 4 + 0 * 128,  5 * 4 + 0 * 128,  9 * 4 + 0 * 128, 13 * 4 + 0 * 128, \
	 17 * 4 + 0 * 128, 21 * 4 + 0 * 128, 25 * 4 + 0 * 128, 29 * 4 + 0 * 128,\
	 2 * 4 + 1 * 128,  6 * 4 + 1 * 128, 10 * 4 + 1 * 128, 14 * 4 + 1 * 128, \
	18 * 4 + 1 * 128, 22 * 4 + 1 * 128, 26 * 4 + 1 * 128, 30 * 4 + 1 * 128, \
	 3 * 4 + 2 * 128,  7 * 4 + 2 * 128, 11 * 4 + 2 * 128, 15 * 4 + 2 * 128, \
	19 * 4 + 2 * 128, 23 * 4 + 2 * 128, 27 * 4 + 2 * 128, 31 * 4 + 2 * 128, \
	 0 * 4 + 3 * 128,  4 * 4 + 3 * 128,  8 * 4 + 3 * 128, 12 * 4 + 3 * 128, \
	16 * 4 + 3 * 128, 20 * 4 + 3 * 128, 24 * 4 + 3 * 128, 28 * 4 + 3 * 128, \
};
//ת�����ұ�2
uint32_t matrix_table_two[32] = {
	 0 * 4 + 2 * 128,  1 * 4 + 2 * 128,  2 * 4 + 2 * 128,  3 * 4 + 2 * 128, \
	 4 * 4 + 2 * 128,  5 * 4 + 2 * 128,  6 * 4 + 2 * 128,  7 * 4 + 2 * 128, \
	 8 * 4 + 3 * 128,  9 * 4 + 3 * 128, 10 * 4 + 3 * 128, 11 * 4 + 3 * 128, \
	12 * 4 + 3 * 128, 13 * 4 + 3 * 128, 14 * 4 + 3 * 128, 15 * 4 + 3 * 128, \
	16 * 4 + 0 * 128, 17 * 4 + 0 * 128, 18 * 4 + 0 * 128, 19 * 4 + 0 * 128, \
	20 * 4 + 0 * 128, 21 * 4 + 0 * 128, 22 * 4 + 0 * 128, 23 * 4 + 0 * 128, \
	24 * 4 + 1 * 128, 25 * 4 + 1 * 128, 26 * 4 + 1 * 128, 27 * 4 + 1 * 128, \
	28 * 4 + 1 * 128, 29 * 4 + 1 * 128, 30 * 4 + 1 * 128, 31 * 4 + 1 * 128, \
};
uint32_t linear_table_two[32] = {
	 2 * 4 + 0 * 128,  6 * 4 + 0 * 128, 10 * 4 + 0 * 128, 14 * 4 + 0 * 128, \
	 18 * 4 + 0 * 128, 22 * 4 + 0 * 128, 26 * 4 + 0 * 128, 30 * 4 + 0 * 128,\
	 3 * 4 + 1 * 128,  7 * 4 + 1 * 128, 11 * 4 + 1 * 128, 15 * 4 + 1 * 128, \
	19 * 4 + 1 * 128, 23 * 4 + 1 * 128, 27 * 4 + 1 * 128, 31 * 4 + 1 * 128, \
	 0 * 4 + 2 * 128,  4 * 4 + 2 * 128,  8 * 4 + 2 * 128, 12 * 4 + 2 * 128, \
	16 * 4 + 2 * 128, 20 * 4 + 2 * 128, 24 * 4 + 2 * 128, 28 * 4 + 2 * 128, \
	 1 * 4 + 3 * 128,  5 * 4 + 3 * 128,  9 * 4 + 3 * 128, 13 * 4 + 3 * 128, \
	17 * 4 + 3 * 128, 21 * 4 + 3 * 128, 25 * 4 + 3 * 128, 29 * 4 + 3 * 128, \
};
//ת�����ұ�3
uint32_t matrix_table_three[32] = {
	 0 * 4 + 3 * 128,  1 * 4 + 3 * 128,  2 * 4 + 3 * 128,  3 * 4 + 3 * 128, \
	 4 * 4 + 3 * 128,  5 * 4 + 3 * 128,  6 * 4 + 3 * 128,  7 * 4 + 3 * 128, \
	 8 * 4 + 0 * 128,  9 * 4 + 0 * 128, 10 * 4 + 0 * 128, 11 * 4 + 0 * 128, \
	12 * 4 + 0 * 128, 13 * 4 + 0 * 128, 14 * 4 + 0 * 128, 15 * 4 + 0 * 128, \
	16 * 4 + 1 * 128, 17 * 4 + 1 * 128, 18 * 4 + 1 * 128, 19 * 4 + 1 * 128, \
	20 * 4 + 1 * 128, 21 * 4 + 1 * 128, 22 * 4 + 1 * 128, 23 * 4 + 1 * 128, \
	24 * 4 + 2 * 128, 25 * 4 + 2 * 128, 26 * 4 + 2 * 128, 27 * 4 + 2 * 128, \
	28 * 4 + 2 * 128, 29 * 4 + 2 * 128, 30 * 4 + 2 * 128, 31 * 4 + 2 * 128, \
};
uint32_t linear_table_three[32] = {
	 3 * 4 + 0 * 128,  7 * 4 + 0 * 128, 11 * 4 + 0 * 128, 15 * 4 + 0 * 128, \
	 19 * 4 + 0 * 128, 23 * 4 + 0 * 128, 27 * 4 + 0 * 128, 31 * 4 + 0 * 128,\
	 0 * 4 + 1 * 128,  4 * 4 + 1 * 128,  8 * 4 + 1 * 128, 12 * 4 + 1 * 128, \
	16 * 4 + 1 * 128, 20 * 4 + 1 * 128, 24 * 4 + 1 * 128, 28 * 4 + 1 * 128, \
	 1 * 4 + 2 * 128,  5 * 4 + 2 * 128,  9 * 4 + 2 * 128, 13 * 4 + 2 * 128, \
	17 * 4 + 2 * 128, 21 * 4 + 2 * 128, 25 * 4 + 2 * 128, 29 * 4 + 2 * 128, \
	 2 * 4 + 3 * 128,  6 * 4 + 3 * 128, 10 * 4 + 3 * 128, 14 * 4 + 3 * 128, \
	18 * 4 + 3 * 128, 22 * 4 + 3 * 128, 26 * 4 + 3 * 128, 30 * 4 + 3 * 128, \
};

//ÿ���߳̿鹲��IV, SK, ency0, lenAC
__constant__ uint8_t constant_iv[12];
__constant__ uint32_t constant_sk[32];
__constant__ uint8_t  constant_ency0[16];
__constant__ uint8_t  constant_lenAC[16];

void otherT(uint8_t T[16][256][16])
{
	int i = 0, j = 0, k = 0;
	uint64_t vh, vl;
	uint64_t zh, zl;
	for (i = 0; i < 256; i++)
	{
		vh = ((uint64_t)T[0][i][0] << 56) ^ ((uint64_t)T[0][i][1] << 48) ^ \
			((uint64_t)T[0][i][2] << 40) ^ ((uint64_t)T[0][i][3] << 32) ^ \
			((uint64_t)T[0][i][4] << 24) ^ ((uint64_t)T[0][i][5] << 16) ^ \
			((uint64_t)T[0][i][6] << 8) ^ ((uint64_t)T[0][i][7]);

		vl = ((uint64_t)T[0][i][8] << 56) ^ ((uint64_t)T[0][i][9] << 48) ^ \
			((uint64_t)T[0][i][10] << 40) ^ ((uint64_t)T[0][i][11] << 32) ^ \
			((uint64_t)T[0][i][12] << 24) ^ ((uint64_t)T[0][i][13] << 16) ^ \
			((uint64_t)T[0][i][14] << 8) ^ ((uint64_t)T[0][i][15]);

		zh = zl = 0;

		for (j = 0; j <= 120; j++)
		{
			if ((j > 0) && (0 == j % 8))
			{
				zh ^= vh;
				zl ^= vl;
				for (k = 1; k <= 16 / 2; k++)
				{
					T[j / 8][i][16 / 2 - k] = (uint8_t)zh;
					zh = zh >> 8;
					T[j / 8][i][16 - k] = (uint8_t)zl;
					zl = zl >> 8;
				}
				zh = zl = 0;
			}
			if (vl & 0x1)
			{
				vl = vl >> 1;
				if (vh & 0x1) { vl ^= 0x8000000000000000; }
				vh = vh >> 1;
				vh ^= 0xe100000000000000;
			}
			else
			{
				vl = vl >> 1;
				if (vh & 0x1) { vl ^= 0x8000000000000000; }
				vh = vh >> 1;
			}
		}
	}
}

//����GF�˷���
void computeTable(uint8_t T[16][256][16], uint8_t H[16])
{
	// zh is the higher 64-bit, zl is the lower 64-bit
	uint64_t zh = 0, zl = 0;
	// vh is the higher 64-bit, vl is the lower 64-bit
	uint64_t vh = ((uint64_t)H[0] << 56) ^ ((uint64_t)H[1] << 48) ^ \
		((uint64_t)H[2] << 40) ^ ((uint64_t)H[3] << 32) ^ \
		((uint64_t)H[4] << 24) ^ ((uint64_t)H[5] << 16) ^ \
		((uint64_t)H[6] << 8) ^ ((uint64_t)H[7]);

	uint64_t vl = ((uint64_t)H[8] << 56) ^ ((uint64_t)H[9] << 48) ^ \
		((uint64_t)H[10] << 40) ^ ((uint64_t)H[11] << 32) ^ \
		((uint64_t)H[12] << 24) ^ ((uint64_t)H[13] << 16) ^ \
		((uint64_t)H[14] << 8) ^ ((uint64_t)H[15]);

	uint8_t temph;

	uint64_t tempvh = vh;
	uint64_t tempvl = vl;
	int i = 0, j = 0;
	for (i = 0; i < 256; i++)
	{
		temph = (uint8_t)i;
		vh = tempvh;
		vl = tempvl;
		zh = zl = 0;

		for (j = 0; j < 8; j++)
		{
			if (0x80 & temph)
			{
				zh ^= vh;
				zl ^= vl;
			}
			if (vl & 0x1)
			{
				vl = vl >> 1;
				if (vh & 0x1) { vl ^= 0x8000000000000000; }
				vh = vh >> 1;
				vh ^= 0xe100000000000000;
			}
			else
			{
				vl = vl >> 1;
				if (vh & 0x1) { vl ^= 0x8000000000000000; }
				vh = vh >> 1;
			}
			temph = temph << 1;
		}
		// get result
		for (j = 1; j <= 16 / 2; j++)
		{
			T[0][i][16 / 2 - j] = (uint8_t)zh;
			zh = zh >> 8;
			T[0][i][16 - j] = (uint8_t)zl;
			zl = zl >> 8;
		}
	}
	otherT(T);
}

/**
 * return the value of (output.H) by looking up tables
 */
void multi(uint8_t T[16][256][16], uint8_t *output)
{
	uint8_t i, j;
	uint8_t temp[16];
	for (i = 0; i < 16; i++)
	{
		temp[i] = output[i];
		output[i] = 0;
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 16; j++)
		{
			output[j] ^= T[i][*(temp + i)][j];
		}
	}
}

/*
 * a: additional authenticated data
 * c: the cipher text or initial vector
 */
void ghash(uint8_t T[16][256][16], uint8_t *add, size_t add_len, uint8_t *cipher, size_t length, uint8_t *output)
{
	/* x0 = 0 */
	*(uint64_t *)output = 0;
	*((uint64_t *)output + 1) = 0;

	/* compute with add */
	int i = 0;
	for (i = 0; i < add_len / 16; i++)
	{
		*(uint64_t *)output ^= *(uint64_t *)add;
		*((uint64_t *)output + 1) ^= *((uint64_t *)add + 1);
		add += 16;
		multi(T, output);
	}

	if (add_len % 16)
	{
		// the remaining add
		for (i = 0; i < add_len % 16; i++)
		{
			*(output + i) ^= *(add + i);
		}
		multi(T, output);
	}

	/* compute with cipher text */
	for (i = 0; i < length / 16; i++)
	{
		*(uint64_t *)output ^= *(uint64_t *)cipher;
		*((uint64_t *)output + 1) ^= *((uint64_t *)cipher + 1);
		cipher += 16;
		multi(T, output);
	}
	if (length % 16)
	{
		// the remaining cipher
		for (i = 0; i < length % 16; i++)
		{
			*(output + i) ^= *(cipher + i);
		}
		multi(T, output);
	}

	/* eor (len(A)||len(C)) */
	uint64_t temp_len = (uint64_t)(add_len * 8); // len(A) = (uint64_t)(add_len*8)
	for (i = 1; i <= 16 / 2; i++)
	{
		output[16 / 2 - i] ^= (uint8_t)temp_len;
		temp_len = temp_len >> 8;
	}
	temp_len = (uint64_t)(length * 8); // len(C) = (uint64_t)(length*8)
	for (i = 1; i <= 16 / 2; i++)
	{
		output[16 - i] ^= (uint8_t)temp_len;
		temp_len = temp_len >> 8;
	}
	multi(T, output);
}

/*
**	���˺��������������Դ洢ģʽת��Ϊ���δ洢ģʽ
**	dev_linear�����Դ洢ģʽ���ݿ飬�����ݿ���������ʽ�洢��ȫ���ڴ���
**	dev_matrix�����δ洢ģʽ���ݿ飬�����ݿ��Ծ�����ʽ�洢��ȫ���ڴ���
*/
__global__ void kernal_linear_to_matrix(\
	uint32_t dev_matrix_table_zero[32], uint32_t dev_linear_table_zero[32], \
	uint32_t dev_matrix_table_one[32], uint32_t dev_linear_table_one[32], \
	uint32_t dev_matrix_table_two[32], uint32_t dev_linear_table_two[32], \
	uint32_t dev_matrix_table_three[32], uint32_t dev_linear_table_three[32], \
	uint8_t dev_linear[PARTICLE_SIZE / STREAM_SIZE], \
	uint8_t dev_matrix[PARTICLE_SIZE / STREAM_SIZE])
{
	__shared__ uint8_t smem[16 * BLOCK_SIZE * 2];
	uint8_t *matrix = smem;
	uint8_t *linear = smem + 16 * BLOCK_SIZE;

	uint32_t dev_offset = blockIdx.x * blockDim.x * 16 + threadIdx.x * 4;
	uint32_t share_offset = threadIdx.x * 4;

	//�Զ���ϲ��ô�ķ�ʽ�����ݴ�ȫ���ڴ滺�浽�����ڴ�
	{
		uint32_t *read = (uint32_t *)(dev_linear + dev_offset);
		uint32_t *write = (uint32_t *)(linear + share_offset);

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(write + i * BLOCK_SIZE) = *(read + i * BLOCK_SIZE);
		}
	}

	//ͬ����
	__syncthreads();

	//���ת��
	{
		uint32_t warpaddr = (threadIdx.x / 32) * (32 * 16);
		uint32_t inertid = threadIdx.x % 32;
		uint32_t *read;
		uint32_t *write;

		//��0��ת��
		write = (uint32_t *)(matrix + warpaddr + dev_matrix_table_zero[inertid]);
		read = (uint32_t *)(linear + warpaddr + dev_linear_table_zero[inertid]);
		*write = *read;

		//��1��ת��
		write = (uint32_t *)(matrix + warpaddr + dev_matrix_table_one[inertid]);
		read = (uint32_t *)(linear + warpaddr + dev_linear_table_one[inertid]);
		*write = *read;

		//��2��ת��
		write = (uint32_t *)(matrix + warpaddr + dev_matrix_table_two[inertid]);
		read = (uint32_t *)(linear + warpaddr + dev_linear_table_two[inertid]);
		*write = *read;

		//��3��ת��
		write = (uint32_t *)(matrix + warpaddr + dev_matrix_table_three[inertid]);
		read = (uint32_t *)(linear + warpaddr + dev_linear_table_three[inertid]);
		*write = *read;
	}

	//ͬ����
	__syncthreads();

	//�Զ���ϲ��ô�ķ�ʽ�����ݴӹ����ڴ�д��ȫ���ڴ�
	{
		uint32_t *write = (uint32_t *)(dev_matrix + dev_offset);
		uint32_t *read = (uint32_t *)(matrix + share_offset);

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(write + i * BLOCK_SIZE) = *(read + i * BLOCK_SIZE);
		}
	}
}

/*
**	���˺����������ɾ��δ洢ģʽת��Ϊ���Դ洢ģʽ
**	dev_matrix�����δ洢ģʽ���ݿ飬�����ݿ��Ծ�����ʽ�洢��ȫ���ڴ���
**	dev_linear�����Դ洢ģʽ���ݿ飬�����ݿ���������ʽ�洢��ȫ���ڴ���
*/
__global__ void kernal_matrix_to_linear(\
	uint32_t dev_matrix_table_zero[32], uint32_t dev_linear_table_zero[32], \
	uint32_t dev_matrix_table_one[32], uint32_t dev_linear_table_one[32], \
	uint32_t dev_matrix_table_two[32], uint32_t dev_linear_table_two[32], \
	uint32_t dev_matrix_table_three[32], uint32_t dev_linear_table_three[32], \
	uint8_t dev_matrix[PARTICLE_SIZE / STREAM_SIZE], \
	uint8_t dev_linear[PARTICLE_SIZE / STREAM_SIZE])
{
	__shared__ uint8_t smem[16 * BLOCK_SIZE * 2];
	uint8_t *matrix = smem;
	uint8_t *linear = smem + 16 * BLOCK_SIZE;
	uint32_t dev_offset = blockIdx.x * blockDim.x * 16 + threadIdx.x * 4;
	uint32_t share_offset = threadIdx.x * 4;

	//�Զ���ϲ��ô�ķ�ʽ�����ݴ�ȫ���ڴ滺�浽�����ڴ�
	{
		uint32_t *read = (uint32_t *)(dev_matrix + dev_offset);
		uint32_t *write = (uint32_t *)(matrix + share_offset);

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(write + i * BLOCK_SIZE) = *(read + i * BLOCK_SIZE);
		}
	}

	//ͬ����
	__syncthreads();

	//���ת��
	{
		uint32_t warpaddr = (threadIdx.x / 32) * (32 * 16);
		uint32_t inertid = threadIdx.x % 32;
		uint32_t *read;
		uint32_t *write;

		//��0��ת��
		read = (uint32_t *)(matrix + warpaddr + dev_matrix_table_zero[inertid]);
		write = (uint32_t *)(linear + warpaddr + dev_linear_table_zero[inertid]);
		*write = *read;

		//��1��ת��
		read = (uint32_t *)(matrix + warpaddr + dev_matrix_table_one[inertid]);
		write = (uint32_t *)(linear + warpaddr + dev_linear_table_one[inertid]);
		*write = *read;

		//��2��ת��
		read = (uint32_t *)(matrix + warpaddr + dev_matrix_table_two[inertid]);
		write = (uint32_t *)(linear + warpaddr + dev_linear_table_two[inertid]);
		*write = *read;

		//��3��ת��
		read = (uint32_t *)(matrix + warpaddr + dev_matrix_table_three[inertid]);
		write = (uint32_t *)(linear + warpaddr + dev_linear_table_three[inertid]);
		*write = *read;
	}

	//ͬ����
	__syncthreads();

	//�Զ���ϲ��ô�ķ�ʽ�����ݴӹ����ڴ�д��ȫ���ڴ�
	{
		uint32_t *write = (uint32_t *)(dev_linear + dev_offset);
		uint32_t *read = (uint32_t *)(linear + share_offset);

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(write + i * BLOCK_SIZE) = *(read + i * BLOCK_SIZE);
		}
	}
}

/*
**	�����㷨�˺��������SM4-CTRģʽ���ܣ�ÿ���̼߳���һ����ţ�֮�����������ݿ������������
**	dev_SboxTable:S��
**	counter:���ݿ����
**	streamid:��ID
**	dev_input:������������
**	dev_output:�����������
*/
__global__ void kernal_enc(uint8_t *const __restrict__ dev_SboxTable, \
	uint32_t dev_matrix_table_zero[32], uint32_t dev_linear_table_zero[32], \
	uint32_t dev_matrix_table_one[32], uint32_t dev_linear_table_one[32], \
	uint32_t dev_matrix_table_two[32], uint32_t dev_linear_table_two[32], \
	uint32_t dev_matrix_table_three[32], uint32_t dev_linear_table_three[32], \
	uint32_t counter, uint32_t streamid, \
	uint8_t dev_input[PARTICLE_SIZE / STREAM_SIZE], \
	uint8_t dev_output[PARTICLE_SIZE / STREAM_SIZE])
{
	__shared__ uint8_t smem[16 * BLOCK_SIZE * 2];
	uint8_t *matrix = smem;
	uint8_t *linear = smem + 16 * BLOCK_SIZE;
	uint8_t *rw_matrix = matrix + (threadIdx.x / 32) * (16 * 32) + (threadIdx.x % 32) * 4;
	uint32_t dev_offset = blockIdx.x * blockDim.x * 16 + threadIdx.x * 4;
	uint32_t share_offset = threadIdx.x * 4;

	{
		uint32_t ulbuf[5];

		{
			//�����̶߳�ȡiv
			uint8_t tidCTR[16];

			*(uint32_t *)(tidCTR + 0) = *(uint32_t *)(constant_iv + 0);
			*(uint32_t *)(tidCTR + 4) = *(uint32_t *)(constant_iv + 4);
			*(uint32_t *)(tidCTR + 8) = *(uint32_t *)(constant_iv + 8);

			*(uint32_t *)(tidCTR + 12) = counter + (uint32_t)(threadIdx.x + blockIdx.x * blockDim.x + streamid * (PARTICLE_SIZE / STREAM_SIZE / 16));
			//*(uint32_t *)(tidCTR + 12) = counter;

			#pragma unroll 4
			for (int i = 0; i < 4; i++)
			{
				ulbuf[i] = (((uint32_t)tidCTR[i * 4]) << 24) | \
					(((uint32_t)tidCTR[i * 4 + 1]) << 16) | \
					(((uint32_t)tidCTR[i * 4 + 2]) << 8) | \
					(uint32_t)tidCTR[i * 4 + 3];
			}
		}

		//32�ֵ�������
		{
			uint32_t temp;
			uint8_t a[4];
			uint32_t bb;

			#pragma unroll 32
			for (int i = 0; i < 32; i++)
			{
				temp = ulbuf[(i + 1) % 5] ^ ulbuf[(i + 2) % 5] ^ ulbuf[(i + 3) % 5] ^ constant_sk[i];
				a[0] = (uint8_t)(temp >> 24);
				a[1] = (uint8_t)(temp >> 16);
				a[2] = (uint8_t)(temp >> 8);
				a[3] = (uint8_t)temp;
				a[0] = dev_SboxTable[a[0]];
				a[1] = dev_SboxTable[a[1]];
				a[2] = dev_SboxTable[a[2]];
				a[3] = dev_SboxTable[a[3]];
				bb = (((uint32_t)a[0]) << 24) | (((uint32_t)a[1]) << 16) | (((uint32_t)a[2]) << 8) | (uint32_t)a[3];
				bb = bb ^ ((bb << 2) | (bb >> 30)) ^ ((bb << 10) | (bb >> 22)) ^ ((bb << 18) | (bb >> 14)) ^ ((bb << 24) | (bb >> 8));
				ulbuf[(i + 4) % 5] = ulbuf[(i + 0) % 5] ^ bb;
			}
		}

		{
			//��д���߳����������ʼ��ַ(���δ洢ģʽ)�����Ĵ���ڹ����ڴ�
			uint8_t temp[4];
			uint8_t *write = rw_matrix;

			temp[0] = (uint8_t)(ulbuf[0] >> 24);
			temp[1] = (uint8_t)(ulbuf[0] >> 16);
			temp[2] = (uint8_t)(ulbuf[0] >> 8);
			temp[3] = (uint8_t)ulbuf[0];
			*(uint32_t *)(rw_matrix + 0 * 128) = *(uint32_t *)temp;

			temp[0] = (uint8_t)(ulbuf[4] >> 24);
			temp[1] = (uint8_t)(ulbuf[4] >> 16);
			temp[2] = (uint8_t)(ulbuf[4] >> 8);
			temp[3] = (uint8_t)ulbuf[4];
			*(uint32_t *)(rw_matrix + 1 * 128) = *(uint32_t *)temp;

			temp[0] = (uint8_t)(ulbuf[3] >> 24);
			temp[1] = (uint8_t)(ulbuf[3] >> 16);
			temp[2] = (uint8_t)(ulbuf[3] >> 8);
			temp[3] = (uint8_t)ulbuf[3];
			*(uint32_t *)(rw_matrix + 2 * 128) = *(uint32_t *)temp;

			temp[0] = (uint8_t)(ulbuf[2] >> 24);
			temp[1] = (uint8_t)(ulbuf[2] >> 16);
			temp[2] = (uint8_t)(ulbuf[2] >> 8);
			temp[3] = (uint8_t)ulbuf[2];
			*(uint32_t *)(rw_matrix + 3 * 128) = *(uint32_t *)temp;
		}
	}

	//ͬ����
	__syncthreads();

	//�������ڴ��о��δ洢ģʽ������ת�������Դ洢ģʽ
	{
		uint32_t warpaddr = (threadIdx.x / 32) * (32 * 16);
		uint32_t inertid = threadIdx.x % 32;
		uint32_t *read;
		uint32_t *write;

		//��0��ת��
		read = (uint32_t *)(matrix + warpaddr + dev_matrix_table_zero[inertid]);
		write = (uint32_t *)(linear + warpaddr + dev_linear_table_zero[inertid]);
		*write = *read;

		//��1��ת��
		read = (uint32_t *)(matrix + warpaddr + dev_matrix_table_one[inertid]);
		write = (uint32_t *)(linear + warpaddr + dev_linear_table_one[inertid]);
		*write = *read;

		//��2��ת��
		read = (uint32_t *)(matrix + warpaddr + dev_matrix_table_two[inertid]);
		write = (uint32_t *)(linear + warpaddr + dev_linear_table_two[inertid]);
		*write = *read;

		//��3��ת��
		read = (uint32_t *)(matrix + warpaddr + dev_matrix_table_three[inertid]);
		write = (uint32_t *)(linear + warpaddr + dev_linear_table_three[inertid]);
		*write = *read;
	}

	//ͬ����
	__syncthreads();

	//�Զ���ϲ��ô�ķ�ʽ��ȡ���ģ������ܺ����������������������ģ��������ĺ����Զ���ϲ��ô�ķ�ʽд��ȫ���ڴ�
	{
		uint32_t *read = (uint32_t *)(dev_input + dev_offset);
		uint32_t *write = (uint32_t *)(dev_output + dev_offset);
		uint32_t *cipher = (uint32_t *)(linear + share_offset);

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(write + i * BLOCK_SIZE) = (*(read + i * BLOCK_SIZE)) ^ (*(cipher + i * BLOCK_SIZE));
		}
	}
}
/*
**	������˷��ӷ�����˺���
**	dev_gfmult_table:������˷���
**	dev_cipher:��������(���δ洢ģʽ)
**	dev_gfmult:������˷����(���δ洢ģʽ)
*/
__global__ void kernal_gfmult(\
	uint8_t dev_gfmult_table[16][256][16], \
	uint8_t dev_cipher[PARTICLE_SIZE / STREAM_SIZE], \
	uint8_t dev_gfmult[PARTICLE_SIZE / STREAM_SIZE])
{
	__shared__ uint8_t smem[16 * BLOCK_SIZE];
	uint8_t *matrix = smem;

	uint32_t dev_offset = blockIdx.x * blockDim.x * 16 + threadIdx.x * 4;
	uint32_t share_offset = threadIdx.x * 4;

	//�Զ���ϲ��ô�ķ�ʽ��ȫ���ڴ��ȡ��������һ��������˷����������������Ľ��д�������ڴ�
	//��ʱ�����ڴ��е����ݿ��Ծ��δ洢ģʽ�洢��
	{
		uint32_t *read_cipher = (uint32_t *)(dev_cipher + dev_offset);
		uint32_t *read_gfmult = (uint32_t *)(dev_gfmult + dev_offset);
		uint32_t *write = (uint32_t *)(matrix + share_offset);

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(write + i * BLOCK_SIZE) = (*(read_cipher + i * BLOCK_SIZE)) ^ (*(read_gfmult + i * BLOCK_SIZE));
		}
	}

	//ͬ����
	__syncthreads();

	//������˷�
	{
		uint8_t *tid_cipher = matrix + (threadIdx.x / 32) * (16 * 32) + (threadIdx.x % 32) * 4;
		uint8_t temp;
		uint8_t *read;

		//�ݴ�GF�˷����
		uint8_t tid_gfmult[16];
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(tid_gfmult + i * 4) = 0;
		}

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			read = tid_cipher + i * (32 * 4);

			#pragma unroll 4
			for (int j = 0; j < 4; j++)
			{
				temp = read[j];

				#pragma unroll 16
				for (int k = 0; k < 16; k++)
				{
					tid_gfmult[k] ^= dev_gfmult_table[i * 4 + j][temp][k];
				}
			}
		}

		//�������ݿ��������˷��Ľ��д�ع����ڴ�
		{
			uint32_t *write = (uint32_t *)(matrix + (threadIdx.x / 32) * (16 * 32) + (threadIdx.x % 32) * 4);

			#pragma unroll 4
			for (int i = 0; i < 4; i++)
			{
				*(write + i * 32) = *(uint32_t *)(tid_gfmult + i * 4);
			}
		}
	}

	//ͬ����
	__syncthreads();

	//�Զ���ϲ��ô�ķ�ʽ�������ڴ��еĳ˷����д��ȫ���ڴ棬��ʱ���ݿ��Ծ��δ洢ģʽ�����ȫ���ڴ��С�
	{
		uint32_t *write = (uint32_t *)(dev_gfmult + dev_offset);
		uint32_t *read = (uint32_t *)(matrix + share_offset);

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(write + i * BLOCK_SIZE) = *(read + i * BLOCK_SIZE);
		}
	}
}

/*
**	���˺�����ɼ���ÿ���߳����յ�GHASH���
**	dev_gfmult_table;������˷���
**	dev_gfmult:������˷����
*/
__global__ void kernal_final(\
	uint8_t dev_gfmult_table[16][256][16], \
	uint32_t dev_matrix_table_zero[32], uint32_t dev_linear_table_zero[32], \
	uint32_t dev_matrix_table_one[32], uint32_t dev_linear_table_one[32], \
	uint32_t dev_matrix_table_two[32], uint32_t dev_linear_table_two[32], \
	uint32_t dev_matrix_table_three[32], uint32_t dev_linear_table_three[32], \
	uint8_t dev_gfmult[PARTICLE_SIZE / STREAM_SIZE])
{
	__shared__ uint8_t smem[16 * BLOCK_SIZE];
	uint8_t *matrix = smem;
	uint32_t dev_offset = blockIdx.x * blockDim.x * 16 + threadIdx.x * 4;
	uint32_t share_offset = threadIdx.x * 4;
	//�Զ���ϲ��ô淽ʽ��ȡǰһ��GF�˷�����������ڴ�
	{
		uint32_t *read = (uint32_t *)(dev_gfmult + dev_offset);
		uint32_t *write = (uint32_t *)(matrix + share_offset);

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(write + i * BLOCK_SIZE) = *(read + i * BLOCK_SIZE);
		}
	}

	//ͬ����
	__syncthreads();

	{
		uint8_t *tid_cipher = matrix + (threadIdx.x / 32) * (16 * 32) + (threadIdx.x % 32) * 4;
		uint8_t temp;
		uint8_t *read;

		//�ݴ�GF�˷��м���
		uint8_t tid_gfmult[16];
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(tid_gfmult + i * 4) = 0;
		}

		//�������������˷�
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			read = tid_cipher + i * (32 * 4);

			#pragma unroll 4
			for (int j = 0; j < 4; j++)
			{
				temp = read[j] ^ constant_lenAC[i * 4 + j];

				#pragma unroll 16
				for (int k = 0; k < 16; k++)
				{
					tid_gfmult[k] ^= dev_gfmult_table[i * 4 + j][temp][k];
				}
			}
		}

		//ÿ���߳���ency0����������յ�tag
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(tid_gfmult + i * 4) ^= *(uint32_t *)(constant_ency0 + i * 4);
		}

		//�Զ���ϲ��ô�ķ�ʽ������GF�˷����д�ع����ڴ棬��ʱ���ݿ��Ծ��δ洢ģʽ����ڹ����ڴ���
		{
			uint32_t *write = (uint32_t *)(matrix + (threadIdx.x / 32) * (16 * 32) + (threadIdx.x % 32) * 4);

			#pragma unroll 4
			for (int i = 0; i < 4; i++)
			{
				*(write + i * 32) = *(uint32_t *)(tid_gfmult + i * 4);
			}
		}
	}

	//ͬ����
	__syncthreads();

	//�������ڴ��еĳ˷��������ϲ��ô�ķ�ʽд��ȫ���ڴ�
	{
		uint32_t *write = (uint32_t *)(dev_gfmult + dev_offset);
		uint32_t *read = (uint32_t *)(matrix + share_offset);

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(write + i * BLOCK_SIZE) = *(read + i * BLOCK_SIZE);
		}
	}
}


void Init_device_memory(device_memory *way, uint8_t add[16], uint8_t iv[12])
{
	//������
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		cudaStreamCreate(&(way->stream[i]));
	}

	//��ʼ���洢ģʽת�����ұ��ڴ�ռ�
	cudaHostAlloc((void**)&(way->dev_matrix_table_zero), 32 * sizeof(uint32_t), cudaHostAllocDefault);
	cudaMemcpy(way->dev_matrix_table_zero, matrix_table_zero, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	cudaHostAlloc((void**)&(way->dev_linear_table_zero), 32 * sizeof(uint32_t), cudaHostAllocDefault);
	cudaMemcpy(way->dev_linear_table_zero, linear_table_zero, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	cudaHostAlloc((void**)&(way->dev_matrix_table_one), 32 * sizeof(uint32_t), cudaHostAllocDefault);
	cudaMemcpy(way->dev_matrix_table_one, matrix_table_one, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	cudaHostAlloc((void**)&(way->dev_linear_table_one), 32 * sizeof(uint32_t), cudaHostAllocDefault);
	cudaMemcpy(way->dev_linear_table_one, linear_table_one, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	cudaHostAlloc((void**)&(way->dev_matrix_table_two), 32 * sizeof(uint32_t), cudaHostAllocDefault);
	cudaMemcpy(way->dev_matrix_table_two, matrix_table_two, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	cudaHostAlloc((void**)&(way->dev_linear_table_two), 32 * sizeof(uint32_t), cudaHostAllocDefault);
	cudaMemcpy(way->dev_linear_table_two, linear_table_two, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	cudaHostAlloc((void**)&(way->dev_matrix_table_three), 32 * sizeof(uint32_t), cudaHostAllocDefault);
	cudaMemcpy(way->dev_matrix_table_three, matrix_table_three, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	cudaHostAlloc((void**)&(way->dev_linear_table_three), 32 * sizeof(uint32_t), cudaHostAllocDefault);
	cudaMemcpy(way->dev_linear_table_three, linear_table_three, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	//������Կ�����������ڴ�
	cudaMemcpyToSymbol(constant_sk, way->ctx.sk, 32 * sizeof(uint32_t));

	//��ʼ��ÿ���̵߳�IV
	cudaMemcpyToSymbol(constant_iv, iv, 12);
	
	//��ʼ��S���ڴ�ռ�
	cudaHostAlloc((void**)&(way->dev_SboxTable), 256, cudaHostAllocDefault);
	cudaMemcpy(way->dev_SboxTable, SboxTable, 256, cudaMemcpyHostToDevice);

	//������������ռ�
	cudaHostAlloc((void**)&(way->dev_input), PARTICLE_SIZE, cudaHostAllocDefault);

	//������������ռ�
	cudaHostAlloc((void**)&(way->dev_output), PARTICLE_SIZE, cudaHostAllocDefault);

	//����ȫ0���Ŀ�
	uint8_t y0[16];
	uint8_t ency0[16];
	memset(y0, 0, 16);

	//��ency0�����������ڴ�
	sm4_crypt_ecb(&way->ctx, 16, y0, ency0);
	cudaMemcpyToSymbol(constant_ency0, ency0, 16);

	uint8_t gfmult_table[16][256][16];
	//����������˷����ұ�
	computeTable(gfmult_table, ency0);

	//��������˷�������ȫ���ڴ�
	cudaHostAlloc((void**)&(way->dev_gfmult_table), \
		sizeof(gfmult_table), cudaHostAllocDefault);
	cudaMemcpy(way->dev_gfmult_table, gfmult_table, \
		sizeof(gfmult_table), cudaMemcpyHostToDevice);

	//��ʼ��������˷����м���
	uint8_t temp[16];
	memset(temp, 0, 16);

	for (int i = 0; i < 16; i++)
	{
		temp[i] ^= add[i];
	}
	multi(gfmult_table, temp);

	uint8_t *gfmult_init = (uint8_t *)malloc(PARTICLE_SIZE);
	for (int i = 0; i < PARTICLE_SIZE / 16; i++)
	{
		memcpy(gfmult_init + i * 16, temp, 16);
	}

	//��ʼ��������˷�����ռ�
	cudaHostAlloc((void**)&(way->dev_gfmult), \
		PARTICLE_SIZE, cudaHostAllocDefault);

	{
		dim3 grid(GRID_SIZE, 1, 1);
		dim3 block(BLOCK_SIZE, 1, 1);

		for (int i = 0; i < STREAM_SIZE; i++)
		{
			//��������˷���������ȫ���ڴ�
			cudaMemcpyAsync(\
				way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE), \
				gfmult_init + i * (PARTICLE_SIZE / STREAM_SIZE), \
				PARTICLE_SIZE / STREAM_SIZE, \
				cudaMemcpyHostToDevice, way->stream[i]);
		}

		for (int i = 0; i < STREAM_SIZE; i++)
		{
			//��ȫ���ڴ��е�GF�˷���������Դ洢ģʽת���ɾ��δ洢ģʽ
			kernal_linear_to_matrix << < grid, block, 0, way->stream[i] >> > (\
				way->dev_matrix_table_zero, way->dev_linear_table_zero, \
				way->dev_matrix_table_one, way->dev_linear_table_one, \
				way->dev_matrix_table_two, way->dev_linear_table_two, \
				way->dev_matrix_table_three, way->dev_linear_table_three, \
				way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE), \
				way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE));
		}

		for (int i = 0; i < STREAM_SIZE; i++)
		{
			//ͬ����
			cudaStreamSynchronize(way->stream[i]);
		}
	}

	free(gfmult_init);
}

/*
**	�����ӿں���:����������豸�ڴ���ͷŹ�����
*/
void Free_device_memory(device_memory *way)
{
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//ͬ����
		cudaStreamSynchronize(way->stream[i]);
	}

	//�ͷ�ȫ���ڴ�
	cudaFreeHost(way->dev_gfmult_table);
	cudaFreeHost(way->dev_IV);
	cudaFreeHost(way->dev_SboxTable);

	cudaFreeHost(way->dev_matrix_table_zero);
	cudaFreeHost(way->dev_linear_table_zero);
	cudaFreeHost(way->dev_matrix_table_one);
	cudaFreeHost(way->dev_linear_table_one);
	cudaFreeHost(way->dev_matrix_table_two);
	cudaFreeHost(way->dev_linear_table_two);
	cudaFreeHost(way->dev_matrix_table_three);
	cudaFreeHost(way->dev_linear_table_three);

	cudaFreeHost(way->dev_input);
	cudaFreeHost(way->dev_output);
	cudaFreeHost(way->dev_gfmult);

	//�ͷ���
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		cudaStreamDestroy(way->stream[i]);
	}
}

/*
**	��֤���������ӿں���
**	counter:�������
**	input:��������
**	output:�������
*/
void sm4_gcm_enc(device_memory *way, uint32_t counter, uint8_t input[PARTICLE_SIZE], uint8_t output[PARTICLE_SIZE])
{
	dim3 grid(GRID_SIZE, 1, 1);
	dim3 block(BLOCK_SIZE, 1, 1);

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//�����Ĵ������ڴ濽�����豸ȫ���ڴ�
		cudaMemcpyAsync(\
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			PARTICLE_SIZE / STREAM_SIZE, \
			cudaMemcpyHostToDevice, way->stream[i]);
	}

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//���������ݿ���м���
		kernal_enc << < grid, block, 0, way->stream[i] >> > (way->dev_SboxTable, \
			way->dev_matrix_table_zero, way->dev_linear_table_zero, \
			way->dev_matrix_table_one, way->dev_linear_table_one, \
			way->dev_matrix_table_two, way->dev_linear_table_two, \
			way->dev_matrix_table_three, way->dev_linear_table_three, \
			counter, i, \
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE));
	}

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//�����ܺ���������ݿ���豸ȫ���ڴ濽���������ڴ�
		cudaMemcpyAsync(output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			PARTICLE_SIZE / STREAM_SIZE, \
			cudaMemcpyDeviceToHost, way->stream[i]);
	}
/*
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//�����Դ洢ģʽ���������ݿ�ת��Ϊ���δ洢ģʽ
		kernal_linear_to_matrix << < grid, block, 0, way->stream[i] >> > (\
			way->dev_matrix_table_zero, way->dev_linear_table_zero, \
			way->dev_matrix_table_one, way->dev_linear_table_one, \
			way->dev_matrix_table_two, way->dev_linear_table_two, \
			way->dev_matrix_table_three, way->dev_linear_table_three, \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE));
	}

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//����������˷��ͼӷ�����
		kernal_gfmult << < grid, block, 0, way->stream[i] >> > (\
			(uint8_t(*)[256][16])(way->dev_gfmult_table), \
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE));
	}
*/
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//��ͬ��
		cudaStreamSynchronize(way->stream[i]);
	}
}

/*
**	��֤���������ӿں���
**	counter:�������
**	input:��������
**	output:�������
*/
void sm4_gcm_dec(device_memory *way, uint32_t counter, uint8_t input[PARTICLE_SIZE], uint8_t output[PARTICLE_SIZE])
{
	dim3 grid(GRID_SIZE, 1, 1);
	dim3 block(BLOCK_SIZE, 1, 1);

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//�����Ĵ������ڴ濽�����豸ȫ���ڴ�
		cudaMemcpyAsync(\
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			PARTICLE_SIZE / STREAM_SIZE, \
			cudaMemcpyHostToDevice, way->stream[i]);
	}

	//�����Դ洢ģʽ���������ݿ�ת��Ϊ���δ洢ģʽ
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		kernal_linear_to_matrix << < grid, block, 0, way->stream[i] >> > (\
			way->dev_matrix_table_zero, way->dev_linear_table_zero, \
			way->dev_matrix_table_one, way->dev_linear_table_one, \
			way->dev_matrix_table_two, way->dev_linear_table_two, \
			way->dev_matrix_table_three, way->dev_linear_table_three, \
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE));
	}

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//����������˷��ͼӷ�����
		kernal_gfmult << < grid, block, 0, way->stream[i] >> > (\
			(uint8_t(*)[256][16])(way->dev_gfmult_table), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE));
	}

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//���������ݿ���н���
		kernal_enc << < grid, block, 0, way->stream[i] >> > (way->dev_SboxTable, \
			way->dev_matrix_table_zero, way->dev_linear_table_zero, \
			way->dev_matrix_table_one, way->dev_linear_table_one, \
			way->dev_matrix_table_two, way->dev_linear_table_two, \
			way->dev_matrix_table_three, way->dev_linear_table_three, \
			counter, i, \
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE));
	}

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//�����ܺ���������ݿ���豸ȫ���ڴ濽���������ڴ�
		cudaMemcpyAsync(output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			PARTICLE_SIZE / STREAM_SIZE, \
			cudaMemcpyDeviceToHost, way->stream[i]);
	}

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//��ͬ��
		cudaStreamSynchronize(way->stream[i]);
	}
}

/*
**	�������ӿں����������յı�ǩ
**	length:�������ݿ鳤��
**	tag:ֵ�������������ִ����Ͻ�������tag���ڴ�ռ�
*/
void sm4_gcm_final(device_memory *way, uint64_t length, uint8_t tag[PARTICLE_SIZE])
{
	uint8_t temp[16];
	/* eor (len(A)||len(C)) */
	uint64_t temp_len = (uint64_t)(16 * 8); // len(A) = (uint64_t)(add_len*8)
	for (int i = 1; i <= 16 / 2; i++)
	{
		temp[16 / 2 - i] = (uint8_t)temp_len;
		temp_len = temp_len >> 8;
	}
	length = length * 16;
	temp_len = (uint64_t)(length * 8); // len(C) = (uint64_t)(length*8)
	for (int i = 1; i <= 16 / 2; i++)
	{
		temp[16 - i] = (uint8_t)temp_len;
		temp_len = temp_len >> 8;
	}

	//��ʼ��(len(A)||len(C))
	cudaMemcpyToSymbol(constant_lenAC, temp, 16);

	dim3 grid(GRID_SIZE, 1, 1);
	dim3 block(BLOCK_SIZE, 1, 1);

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//�������յ�GHASH���
		kernal_final << < grid, block, 0, way->stream[i] >> > ((uint8_t(*)[256][16])(way->dev_gfmult_table), \
			way->dev_matrix_table_zero, way->dev_linear_table_zero, \
			way->dev_matrix_table_one, way->dev_linear_table_one, \
			way->dev_matrix_table_two, way->dev_linear_table_two, \
			way->dev_matrix_table_three, way->dev_linear_table_three, \
			way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE));
	}

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//��ȫ���ڴ��о��δ洢ģʽ��GHASH���ת�������Դ洢ģʽ
		kernal_matrix_to_linear << < grid, block, 0, way->stream[i] >> > (\
			way->dev_matrix_table_zero, way->dev_linear_table_zero, \
			way->dev_matrix_table_one, way->dev_linear_table_one, \
			way->dev_matrix_table_two, way->dev_linear_table_two, \
			way->dev_matrix_table_three, way->dev_linear_table_three, \
			way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE));
	}

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//��ÿ���̵߳ı�ǩtag��ȫ���ڴ濽���������ڴ�
		cudaMemcpyAsync(tag + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE), \
			PARTICLE_SIZE / STREAM_SIZE, \
			cudaMemcpyDeviceToHost, way->stream[i]);
	}

	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//ͬ����
		cudaStreamSynchronize(way->stream[i]);
	}
}