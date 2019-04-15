#ifndef SM4CUDA_H_
#define SM4CUDA_H_

#define SM4_ENCRYPT     1
#define SM4_DECRYPT     0

//�߳̿��С
#define BLOCK_SIZE		512 
//�����С
#define GRID_SIZE		1024
//������
#define STREAM_SIZE 	4
//�ӽ������� 
#define PARTICLE_SIZE	(16 * BLOCK_SIZE * \
						GRID_SIZE * STREAM_SIZE)


typedef struct
{
	/*
		�ӽ���ģʽ��
		mode = SM4_ENCRYPT : ����
		mode = SM4_DECRYPT : ����
	*/
	int mode;

	/*
		�ӽ��ܲ��õ�����Կ
	*/
	uint32_t sk[32];
}sm4_context;

typedef struct
{
	//ִ����
	cudaStream_t stream[STREAM_SIZE];

	//����Կ
	sm4_context ctx;

	//S��
	uint8_t *dev_SboxTable;

	//ÿ���̶߳�����IV
	uint8_t *dev_IV;

	//ת�����ұ�
	uint32_t *dev_matrix_table_zero;
	uint32_t *dev_linear_table_zero;

	uint32_t *dev_matrix_table_one;
	uint32_t *dev_linear_table_one;

	uint32_t *dev_matrix_table_two;
	uint32_t *dev_linear_table_two;

	uint32_t *dev_matrix_table_three;
	uint32_t *dev_linear_table_three;

	//GF�˷����ұ�
	uint8_t *dev_gfmult_table;

	//��������
	uint8_t *dev_input;

	//�������
	uint8_t *dev_output;

	//GF�˷�������
	uint8_t *dev_gfmult;
}device_memory;

inline void GET_UINT_BE(uint32_t *n, uint8_t *b, uint32_t i);
inline void PUT_UINT_BE(uint32_t n, uint8_t *b, uint32_t i);
inline uint8_t sm4Sbox(uint8_t inch);
inline uint32_t ROTL(uint32_t x, uint32_t n);
inline void SWAP(uint32_t *a, uint32_t *b);
uint32_t sm4Lt(uint32_t ka);
uint32_t sm4F(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t rk);
uint32_t sm4CalciRK(uint32_t ka);
void sm4_setkey(uint32_t SK[32], uint8_t key[16]);
void sm4_one_round(uint32_t sk[32], uint8_t input[16], uint8_t output[16]);

//�û��ӿں���, ����ECBģʽ�����ݳ���Ϊ16Byte��������������16�ֽڵ���Ҫ���0
void sm4_setkey_enc(sm4_context *ctx, uint8_t key[16]);
void sm4_setkey_dec(sm4_context *ctx, uint8_t key[16]);
void sm4_crypt_ecb(sm4_context *ctx, int length, uint8_t *input, uint8_t *output);

//����GF�˷����ұ�
void computeTable(uint8_t T[16][256][16], uint8_t H[16]);
//GF�˷�
void multi(uint8_t T[16][256][16], uint8_t *output);
//GHASH
void ghash(uint8_t T[16][256][16], uint8_t *add, size_t add_len, uint8_t *cipher, size_t length, uint8_t *output);

//��ʼ���ӿں���
void Init_device_memory(device_memory *way, \
	uint8_t add[16], uint8_t iv[12]);
//��֤���ܽӿں���
void sm4_gcm_enc(device_memory *way, uint32_t counter, \
	uint8_t input[PARTICLE_SIZE], uint8_t output[PARTICLE_SIZE]);
//��֤���ܽӿں���
void sm4_gcm_dec(device_memory *way, uint32_t counter, \
	uint8_t input[PARTICLE_SIZE], uint8_t output[PARTICLE_SIZE]);
//�����ǩ�ӿں���
void sm4_gcm_final(device_memory *way, \
	uint64_t length, uint8_t tag[PARTICLE_SIZE]);
//�ͷŴ洢�ռ�ӿں���
void Free_device_memory(device_memory *way);

#endif 
