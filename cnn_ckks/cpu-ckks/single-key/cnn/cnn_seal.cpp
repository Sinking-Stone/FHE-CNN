#include "cnn_seal.h"

#include <fstream>
extern ofstream pFile;

TensorCipher::TensorCipher()
{
    k_=0;
    h_=0;
    w_=0;
	c_=0;
	t_=0;
    p_=0;
}
TensorCipher::TensorCipher(int logn, int k, int h, int w, int c, int t, int p, vector<double> data, Encryptor &encryptor, CKKSEncoder &encoder, int logp)
{
    if(k != 1) throw std::invalid_argument("supported k is only 1 right now");
    
	// 1 <= logn <= 16
    if(logn < 1 || logn > 16) throw std::out_of_range("the value of logn is out of range");
    if(data.size() > static_cast<long unsigned int>(1<<logn)) throw std::out_of_range("the size of data is larger than n");

    this->k_ = k;
    this->h_ = h;
	this->w_ = w;
	this->c_ = c;
    this->t_ = t;
	this->p_ = p;
	this->logn_ = logn;

	// generate vector that contains data
	vector<double> vec;
    for(int i=0; i<static_cast<int>(data.size()); i++) vec.emplace_back(data[i]);
    for(int i=data.size(); i<1<<logn; i++) vec.emplace_back(0);      // zero padding

    // vec size = n
    if(vec.size() != static_cast<long unsigned int>(1<<logn)) throw std::out_of_range("the size of vec is not n");

	// encode & encrypt
	Plaintext plain;
	Ciphertext cipher;
	double scale = pow(2.0, logp);
	encoder.encode(vec, scale, plain);
	encryptor.encrypt(plain, cipher);
	this->set_ciphertext(cipher);

}
TensorCipher::TensorCipher(int logn, int k, int h, int w, int c, int t, int p, Ciphertext cipher)
{
    this->k_ = k;
    this->h_ = h;
	this->w_ = w;
	this->c_ = c;
    this->t_ = t;
	this->p_ = p;
	this->logn_ = logn;
	this->cipher_ = cipher;
}
int TensorCipher::k() const
{
	return k_;
}
int TensorCipher::h() const
{
	return h_;
}
int TensorCipher::w() const
{
	return w_;
}
int TensorCipher::c() const
{
	return c_;
}
int TensorCipher::t() const
{
	return t_;
}
int TensorCipher::p() const
{
	return p_;
}
int TensorCipher::logn() const
{
	return logn_;
}
Ciphertext TensorCipher::cipher() const
{
	return cipher_;
}
void TensorCipher::set_ciphertext(Ciphertext cipher)
{
	cipher_ = cipher;
}
void TensorCipher::print_parms()
{
	cout << "k: " << k_ << endl;
    cout << "h: " << h_ << endl;
    cout << "w: " << w_ << endl;
	cout << "c: " << c_ << endl;
	cout << "t: " << t_ << endl;
	cout << "p: " << p_ << endl;
}

void multiplexed_parallel_convolution_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, int co, int st, int fh, int fw, const vector<double> &data, vector<double> running_var, vector<double> constant_weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, vector<Ciphertext> &cipher_pool, ofstream &output, Decryptor &decryptor, SEALContext &context, size_t stage, bool end)
{
    cout << "卷积multiplexed parallel convolution..." << endl;
    output << "卷积multiplexed parallel convolution..." << endl;
	pFile << "卷积multiplexed parallel convolution..." << endl;
	cout << "卷积remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	cout << "卷积scale: " << cnn_in.cipher().scale() << endl;
	output << "卷积remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	output << "卷积scale: " << cnn_in.cipher().scale() << endl;
	pFile << "卷积remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	pFile << "卷积scale: " << cnn_in.cipher().scale() << endl;
	cout<< "卷积之前的参数大小，分别是：hi、wi、ci、ki、ti、pi、st、fh、fw："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<"、"<<st<<"、"<<fh<<"、"<<fw<<endl;
	output<< "卷积之前的参数大小，分别是：hi、wi、ci、ki、ti、pi、st、fh、fw："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<"、"<<st<<"、"<<fh<<"、"<<fw<<endl;
	pFile<< "卷积之前的参数大小，分别是：hi、wi、ci、ki、ti、pi、st、fh、fw："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<"、"<<st<<"、"<<fh<<"、"<<fw<<endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	// convolution_seal_sparse(cnn_in, cnn_out, hprime, st, kernel, false, data, running_var, constant_weight, epsilon, encoder, encryptor, scale_evaluator, gal_keys, cipher_pool, end);
	multiplexed_parallel_convolution_seal(cnn_in, cnn_out, co, st, fh, fw, data, running_var, constant_weight, epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, end);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// cout << "convolution " << stage << " result" << endl;
	output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	pFile << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// output << "convolution " << stage << " result" << endl;
	// cout<<"卷积之后的数据：\n";
    // decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	// cout<<"卷积之后的参数为：\n ";
	// cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "卷积remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "卷积scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "卷积remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	output << "卷积scale: " << cnn_out.cipher().scale() << endl << endl;
	pFile << "卷积remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	pFile << "卷积scale: " << cnn_out.cipher().scale() << endl;
	cout<< "卷积之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	output<< "卷积之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	pFile<< "卷积之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl << endl;
}
void multiplexed_parallel_batch_norm_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> bias, vector<double> running_mean, vector<double> running_var, vector<double> weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, double B, ofstream &output, Decryptor &decryptor, SEALContext &context, size_t stage, bool end)
{
    cout << "归一化multiplexed parallel batch normalization..." << endl;
    output << "归一化multiplexed parallel batch normalization..." << endl;
	cout << "归一化remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	cout << "归一化scale: " << cnn_in.cipher().scale() << endl;
	output << "归一化remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	output << "归一化scale: " << cnn_in.cipher().scale() << endl;
	pFile << "归一化remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	pFile << "归一化scale: " << cnn_in.cipher().scale() << endl;
	cout<< "归一化之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	output<< "归一化之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	pFile<< "归一化之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	// batch norm
	time_start = chrono::high_resolution_clock::now();
	multiplexed_parallel_batch_norm_seal(cnn_in, cnn_out, bias, running_mean, running_var, weight, epsilon, encoder, encryptor, evaluator, B, end); 
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "batch normalization " << stage << " result" << endl;
	output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	output << "batch normalization " << stage << " result" << endl;
	// cout<<"归一化之后的数据：\n";
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	// cout<<"归一化之后的参数：\n";
	// cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "归一化：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "归一化：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	output << "scale: " << cnn_out.cipher().scale() << endl << endl;
	pFile << "归一化：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	pFile << "scale: " << cnn_out.cipher().scale() << endl;
	cout<< "归一化之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	output<< "归一化之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	pFile<< "归一化之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl << endl;
}
void approx_ReLU_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, long alpha, vector<Tree> &tree, double scaled_val, long scalingfactor, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double B, ofstream &output, SEALContext &context, GaloisKeys &gal_keys, size_t stage)
{
    cout << "激活approximate ReLU..." << endl;
    output << "激活approximate ReLU..." << endl;
	cout << "激活remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	cout << "激活scale: " << cnn_in.cipher().scale() << endl;
	output << "激活remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	output << "激活scale: " << cnn_in.cipher().scale() << endl;
	pFile << "激活remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	pFile << "激活scale: " << cnn_in.cipher().scale() << endl;
	cout<< "激活之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	output<< "激活之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	pFile<< "激活之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;

	cout<<"激活之前的数据：\n";
	output<<"激活之前的数据：\n";
	decrypt_and_print(cnn_in.cipher(), decryptor, encoder, 1<<cnn_in.logn(), 256, 2); 
	decrypt_and_print_txt(cnn_in.cipher(), decryptor, encoder, 1<<cnn_in.logn(), 256, 2,output); 
	// cout<<"激活之前的参数：\n";
	// cnn_in.print_parms();


	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	ReLU_seal(cnn_in, cnn_out, comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, B);
	// ReLU_remove_imaginary_seal(cnn_in, cnn_out, comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, scale_evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, gal_keys, B);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "ReLU function " << stage << " result" << endl;
	output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	output << "ReLU function " << stage << " result" << endl;
	pFile << "time : " << time_diff.count() / 1000 << " ms" << endl;
	pFile << "ReLU function " << stage << " result" << endl;
	cout<<"RELU之后的数据：\n";
	decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<cnn_in.logn(), 256, 2); 
	cout<<"RELU之后的参数：\n";
	cnn_out.print_parms();
	output<<"RELU之后的数据：\n";
	decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<cnn_in.logn(), 256, 2, output);
	cout << "ReLU之后的层级：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "ReLU之后的层级：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	output << "scale: " << cnn_out.cipher().scale() << endl << endl;
	pFile << "ReLU之后的层级：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	pFile << "scale: " << cnn_out.cipher().scale() << endl;
	cout<< "激活之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	output<< "激活之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	pFile<< "激活之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl << endl;

	// cout << "intermediate decrypted values: " << endl;
	// output << "intermediate decrypted values: " << endl;
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 4, 1, output); // cnn_out.print_parms();
}


void adv_Approx_ReLU_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, long alpha, vector<Tree> &tree, double scaled_val, long scalingfactor, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double B, ofstream &output, SEALContext &context, GaloisKeys &gal_keys, size_t stage)
{
    cout << "激活approximate ReLU..." << endl;
    output << "激活approximate ReLU..." << endl;
	cout << "激活remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	cout << "激活scale: " << cnn_in.cipher().scale() << endl;
	output << "激活remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	output << "激活scale: " << cnn_in.cipher().scale() << endl;
	pFile << "激活remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	pFile << "激活scale: " << cnn_in.cipher().scale() << endl;
	cout<< "激活之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	output<< "激活之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	pFile<< "激活之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	adv_ReLU_seal(cnn_in, cnn_out, comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, B);
	// ReLU_remove_imaginary_seal(cnn_in, cnn_out, comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, scale_evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, gal_keys, B);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "ReLU function " << stage << " result" << endl;
	output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	output << "ReLU function " << stage << " result" << endl;
	pFile << "time : " << time_diff.count() / 1000 << " ms" << endl;
	pFile << "ReLU function " << stage << " result" << endl;
	// cout<<"RELU之后的数据：\n";
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	// cout<<"RELU之后的参数：\n";
	// cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "ReLU之后的层级：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "ReLU之后的层级：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	output << "scale: " << cnn_out.cipher().scale() << endl << endl;
	pFile << "ReLU之后的层级：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	pFile << "scale: " << cnn_out.cipher().scale() << endl;
	cout<< "激活之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	output<< "激活之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	pFile<< "激活之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl << endl;

	// cout << "intermediate decrypted values: " << endl;
	// output << "intermediate decrypted values: " << endl;
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 4, 1, output); // cnn_out.print_parms();
}

void adv_Approx_ReLU1_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, long alpha, vector<Tree> &tree, double scaled_val, long scalingfactor, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double B, ofstream &output, SEALContext &context, GaloisKeys &gal_keys, size_t stage)
{
    cout << "激活approximate ReLU..." << endl;
    output << "激活approximate ReLU..." << endl;
	cout << "激活remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	cout << "激活scale: " << cnn_in.cipher().scale() << endl;
	output << "激活remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	output << "激活scale: " << cnn_in.cipher().scale() << endl;
	pFile << "激活remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	pFile << "激活scale: " << cnn_in.cipher().scale() << endl;
	cout<< "激活之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	output<< "激活之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	pFile<< "激活之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	adv_ReLU1_seal(cnn_in, cnn_out, comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, B);
	// ReLU_remove_imaginary_seal(cnn_in, cnn_out, comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, scale_evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, gal_keys, B);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "ReLU function " << stage << " result" << endl;
	output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	output << "ReLU function " << stage << " result" << endl;
	pFile << "time : " << time_diff.count() / 1000 << " ms" << endl;
	pFile << "ReLU function " << stage << " result" << endl;
	// cout<<"RELU之后的数据：\n";
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	// cout<<"RELU之后的参数：\n";
	// cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "ReLU之后的层级：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "ReLU之后的层级：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	output << "scale: " << cnn_out.cipher().scale() << endl << endl;
	pFile << "ReLU之后的层级：remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	pFile << "scale: " << cnn_out.cipher().scale() << endl;
	cout<< "激活之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	output<< "激活之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	pFile<< "激活之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl << endl;

	// cout << "intermediate decrypted values: " << endl;
	// output << "intermediate decrypted values: " << endl;
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 4, 1, output); // cnn_out.print_parms();
}

void bootstrap_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Bootstrapper &bootstrapper, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context, size_t stage)
{
    cout << "自举bootstrapping..." << endl;
    output << "自举bootstrapping..." << endl;
	pFile << "自举bootstrapping..." << endl;
	cout << "自举remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	cout << "自举scale: " << cnn_in.cipher().scale() << endl;
	output << "自举remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	output << "自举scale: " << cnn_in.cipher().scale() << endl;
	pFile << "自举remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	pFile << "自举scale: " << cnn_in.cipher().scale() << endl;
	cout<< "自举之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	output<< "自举之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	pFile<< "自举之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	Ciphertext ctxt, rtn;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	ctxt = cnn_in.cipher();
	time_start = chrono::high_resolution_clock::now();
	// bootstrapper.bootstrap_3(rtn, ctxt);
	bootstrapper.bootstrap_real_3(rtn, ctxt);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	pFile << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cnn_out.set_ciphertext(rtn);
    cout << "bootstrapping " << stage << " result" << endl;
    output << "bootstrapping " << stage << " result" << endl;
	pFile << "bootstrapping " << stage << " result" << endl;
	// cout<<"自举之后的数据：\n";
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	// cout<<"自举之后的参数：\n";
	// cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "自举remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "自举scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "自举remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	output << "自举scale: " << cnn_out.cipher().scale() << endl << endl;
	pFile << "自举remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	pFile << "自举scale: " << cnn_out.cipher().scale() << endl;
	cout<< "自举之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	output<< "自举之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	pFile<< "自举之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl << endl;

}
void cipher_add_seal_print(const TensorCipher &cnn1, const TensorCipher &cnn2, TensorCipher &destination, Evaluator &evaluator, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "残差相加cipher add..." << endl;
    output << "残差相加cipher add..." << endl;
	pFile << "残差相加cipher add..." << endl;

	cout << "残差相加remaining level Cnn1 : " << context.get_context_data(cnn1.cipher().parms_id())->chain_index() << endl;
	cout << "残差相加scale Cnn1 : " << cnn1.cipher().scale() << endl;
	output << "残差相加remaining level Cnn1 : " << context.get_context_data(cnn1.cipher().parms_id())->chain_index() << endl;
	output << "残差相加scale Cnn1 : " << cnn1.cipher().scale() << endl;
	pFile << "残差相加remaining level Cnn1  : " << context.get_context_data(cnn1.cipher().parms_id())->chain_index() << endl;
	pFile << "残差相加scale Cnn1 : " << cnn1.cipher().scale() << endl;

	cout << "残差相加remaining level Cnn2  : " << context.get_context_data(cnn2.cipher().parms_id())->chain_index() << endl;
	cout << "残差相加scale Cnn2 : " << cnn2.cipher().scale() << endl;
	output << "残差相加remaining level Cnn2  : " << context.get_context_data(cnn2.cipher().parms_id())->chain_index() << endl;
	output << "残差相加scale Cnn2 : " << cnn2.cipher().scale() << endl;
	pFile << "残差相加remaining level Cnn2  : " << context.get_context_data(cnn2.cipher().parms_id())->chain_index() << endl;
	pFile << "残差相加scale Cnn2 : " << cnn2.cipher().scale() << endl;
	cout<< "cnn1残差相加之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn1.h()<<"、"<<cnn1.w()<<"、"<<cnn1.c()<<"、"<<cnn1.k()<<"、"<<cnn1.t()<<"、"<<cnn1.p()<<endl;
	output<< "cnn1残差相加之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn1.h()<<"、"<<cnn1.w()<<"、"<<cnn1.c()<<"、"<<cnn1.k()<<"、"<<cnn1.t()<<"、"<<cnn1.p()<<endl;
	pFile<< "cnn1残差相加之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn1.h()<<"、"<<cnn1.w()<<"、"<<cnn1.c()<<"、"<<cnn1.k()<<"、"<<cnn1.t()<<"、"<<cnn1.p()<<endl;

	cout<< "cnn2残差相加之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn2.h()<<"、"<<cnn2.w()<<"、"<<cnn2.c()<<"、"<<cnn2.k()<<"、"<<cnn2.t()<<"、"<<cnn2.p()<<endl;
	output<< "cnn2残差相加之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn2.h()<<"、"<<cnn2.w()<<"、"<<cnn2.c()<<"、"<<cnn2.k()<<"、"<<cnn2.t()<<"、"<<cnn2.p()<<endl;
	pFile<< "cnn2残差相加之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn2.h()<<"、"<<cnn2.w()<<"、"<<cnn2.c()<<"、"<<cnn2.k()<<"、"<<cnn2.t()<<"、"<<cnn2.p()<<endl;

	int logn = cnn1.logn();
	cnn_add_seal(cnn1, cnn2, destination, evaluator);
	// cout<<"残差相加之后的数据：\n";
	// decrypt_and_print(destination.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	// cout<<"残差相加之后的参数：\n";
	// destination.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "残差相加remaining level : " << context.get_context_data(destination.cipher().parms_id())->chain_index() << endl;
	cout << "残差相加scale: " << destination.cipher().scale() << endl << endl;
	output << "残差相加remaining level : " << context.get_context_data(destination.cipher().parms_id())->chain_index() << endl;
	output << "残差相加scale: " << destination.cipher().scale() << endl << endl;
	pFile << "残差相加remaining level : " << context.get_context_data(destination.cipher().parms_id())->chain_index() << endl;
	pFile << "残差相加scale: " << destination.cipher().scale() << endl;
	cout<< "残差相加之后的参数大小，分别是：ho、wo、co、ko、to、po："<<destination.h()<<"、"<<destination.w()<<"、"<<destination.c()<<"、"<<destination.k()<<"、"<<destination.t()<<"、"<<destination.p()<<endl;
	output<< "残差相加之后的参数大小，分别是：ho、wo、co、ko、to、po："<<destination.h()<<"、"<<destination.w()<<"、"<<destination.c()<<"、"<<destination.k()<<"、"<<destination.t()<<"、"<<destination.p()<<endl;
	pFile<< "残差相加之后的参数大小，分别是：ho、wo、co、ko、to、po："<<destination.h()<<"、"<<destination.w()<<"、"<<destination.c()<<"、"<<destination.k()<<"、"<<destination.t()<<"、"<<destination.p()<<endl << endl;
}
void multiplexed_parallel_downsampling_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context, GaloisKeys &gal_keys, ofstream &output)
{
    cout << "下采样multiplexed parallel downsampling..." << endl;
    output << "下采样multiplexed parallel downsampling..." << endl;
	cout << "下采样remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	cout << "下采样scale: " << cnn_in.cipher().scale() << endl << endl;
	output << "下采样remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	output << "下采样scale: " << cnn_in.cipher().scale() << endl << endl;
	pFile << "下采样remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	pFile << "下采样scale: " << cnn_in.cipher().scale() << endl << endl;
	cout<< "下采样之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	output<< "下采样之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	pFile<< "下采样之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;

	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	multiplexed_parallel_downsampling_seal(cnn_in, cnn_out, evaluator, gal_keys);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// cout<<"下采样之后的数据：\n";
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	// cout<<"下采样之后的参数：\n";
	// cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "下采样remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "下采样scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "下采样remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	output << "下采样scale: " << cnn_out.cipher().scale() << endl << endl;
	pFile << "下采样remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	pFile << "下采样scale: " << cnn_out.cipher().scale() << endl;
	cout<< "下采样之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	output<< "下采样之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	pFile<< "下采样之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl << endl;
}
void averagepooling_seal_scale_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, double B, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "平均池化average pooling..." << endl;
    output << "平均池化average pooling..." << endl;
	cout << "平均池化remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	cout << "平均池化scale: " << cnn_in.cipher().scale() << endl << endl;
	output << "平均池化remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	output << "平均池化scale: " << cnn_in.cipher().scale() << endl << endl;
	pFile << "平均池化remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	pFile << "平均池化scale: " << cnn_in.cipher().scale() << endl << endl;
	cout<< "平均池化之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	output<< "平均池化之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	pFile<< "平均池化之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	// averagepooling_seal_scale(cnn_in, cnn_out, scale_evaluator, gal_keys, B);
	averagepooling_seal_scale(cnn_in, cnn_out, evaluator, gal_keys, B, encoder, decryptor, output);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// cout<<"平均池化之后的数据：\n";
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	// cout<<"平均池化之后的参数：\n";
	// cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "平均池化remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "平均池化scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "平均池化remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	output << "平均池化scale: " << cnn_out.cipher().scale() << endl << endl;
	pFile << "平均池化remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	pFile << "平均池化scale: " << cnn_out.cipher().scale() << endl;
	cout<< "平均池化之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	output<< "平均池化之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	pFile<< "平均池化之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl << endl;
}
void fully_connected_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> matrix, vector<double> bias, int q, int r, Evaluator &evaluator, GaloisKeys &gal_keys, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "全连接fully connected layer..." << endl;
    output << "全连接fully connected layer..." << endl;
	cout << "全连接remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	cout << "全连接scale: " << cnn_in.cipher().scale() << endl << endl;
	output << "全连接remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	output << "全连接scale: " << cnn_in.cipher().scale() << endl << endl;
	pFile << "全连接remaining level : " << context.get_context_data(cnn_in.cipher().parms_id())->chain_index() << endl;
	pFile << "全连接scale: " << cnn_in.cipher().scale() << endl;
	cout<< "全连接之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	output<< "全连接之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	pFile<< "全连接之前的参数大小，分别是：hi、wi、ci、ki、ti、pi："<<cnn_in.h()<<"、"<<cnn_in.w()<<"、"<<cnn_in.c()<<"、"<<cnn_in.k()<<"、"<<cnn_in.t()<<"、"<<cnn_in.p()<<endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	matrix_multiplication_seal(cnn_in, cnn_out, matrix, bias, q, r, evaluator, gal_keys);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// cout<<"全连接化之后的数据：\n";
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	// cout<<"全连接化之后的参数：\n";
	// cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "全连接remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "全连接scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "全连接remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	output << "全连接scale: " << cnn_out.cipher().scale() << endl << endl;
	pFile << "全连接remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	pFile << "全连接scale: " << cnn_out.cipher().scale() << endl;
	cout<< "全连接之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	output<< "全连接之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl;
	pFile<< "全连接之后的参数大小，分别是：ho、wo、co、ko、to、po："<<cnn_out.h()<<"、"<<cnn_out.w()<<"、"<<cnn_out.c()<<"、"<<cnn_out.k()<<"、"<<cnn_out.t()<<"、"<<cnn_out.p()<<endl << endl;
}
void multiplexed_parallel_convolution_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, int co, int st, int fh, int fw, const vector<double> &data, vector<double> running_var, vector<double> constant_weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, vector<Ciphertext> &cipher_pool, bool end)
{
	// set parameters 参数中的st表示步长，fh和fw表示卷积核的高和宽，data表示卷积核的值，running_var表示卷积核的方差，constant_weight表示卷积核的权重，epsilon表示方差的偏置
    vector<double> conv_data;
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 0, ho = 0, wo = 0, to = 0, po = 0;

	// error check
	if(st != 1 && st != 2) throw invalid_argument("supported st is only 1 or 2");		// check if st is 1 or 2
    if(static_cast<int>(data.size()) != fh*fw*ci*co) throw std::invalid_argument("the size of data vector is not ker x ker x h x h");	// check if the size of data vector is kernel x kernel x h x h'
	if(log2_long(ki) == -1) throw std::invalid_argument("ki is not power of two");

	if(static_cast<int>(running_var.size())!=co || static_cast<int>(constant_weight.size())!=co) throw std::invalid_argument("the size of running_var or weight is not correct");
	for(auto num : running_var) if(num<pow(10,-16) && num>-pow(10,-16)) throw std::invalid_argument("the size of running_var is too small. nearly zero.");

	// set ho, wo, ko
	if(st == 1) 
	{
		ho = hi;
		wo = wi;
		ko = ki;
	}
	else if(st == 2) 
	{
		if(hi%2 == 1 || wi%2 == 1) throw std::invalid_argument("hi or wi is not even");
		ho = hi/2;
		wo = wi/2;
		ko = 2*ki;
	}

	// set to, po, q
	long n = 1<<logn;
	to = (co+ko*ko-1) / (ko*ko);
	po =  pow2(floor_to_int(log(static_cast<double>(n)/static_cast<double>(ko*ko*ho*wo*to)) / log(2.0))); 	// 这是论文中提到的p
	long q = (co+pi-1)/pi;

	// check if pi, po | n
	if(n%pi != 0) throw std::out_of_range("n is not divisible by pi");
	if(n%po != 0) throw std::out_of_range("n is not divisible by po");

	// check if ki^2 hi wi ti pi <= n and ko^2 ho wo to po <= n
	if(ki*ki*hi*wi*ti*pi > n) throw std::out_of_range("ki^2 hi wi ti pi is larger than n");
	if(ko*ko*ho*wo*to*po > (1<<logn)) throw std::out_of_range("ko^2 ho wo to po is larger than n");

	// cout<<"卷积之前密文参数：\n";
	// pFile << "ko: " << ko << "\tho: " << ho << "\two: " << wo << "\tto: " << to << "\tpo: " << po << "\tq: " << q << "\tpi: " << pi << "\tci: " << ci << "\tco: " << co <<  endl;

	// variable
	vector<vector<vector<vector<double>>>> weight(fh, vector<vector<vector<double>>>(fw, vector<vector<double>>(ci, vector<double>(co, 0.0))));		// weight tensor
	vector<vector<vector<vector<double>>>> compact_weight_vec(fh, vector<vector<vector<double>>>(fw, vector<vector<double>>(q, vector<double>(n, 0.0))));	// multiplexed parallel shifted weight tensor
	vector<vector<vector<vector<double>>>> select_one(co, vector<vector<vector<double>>>(ko*ho, vector<vector<double>>(ko*wo, vector<double>(to, 0.0))));
	vector<vector<double>> select_one_vec(co, vector<double>(1<<logn, 0.0));

	cout<<"weight setting..."<<endl;
	pFile<<"weight setting..."<<endl;
	// weight setting
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			for(int j3=0; j3<ci; j3++)
			{
				for(int j4=0; j4<co; j4++)
				{
					weight[i1][i2][j3][j4] = data[fh*fw*ci*j4 + fh*fw*j3 + fw*i1 + i2];
				}
			}
		}
	}

	cout<<"紧凑compact shifted weight vector setting..."<<endl;
	pFile<<"紧凑compact shifted weight vector setting..."<<endl;
	// compact shifted weight vector setting
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			for(int i9=0; i9<q; i9++)
			{
				for(int j8=0; j8<n; j8++)
				{
					int j5 = ((j8%(n/pi))%(ki*ki*hi*wi))/(ki*wi), j6 = (j8%(n/pi))%(ki*wi), i7 = (j8%(n/pi))/(ki*ki*hi*wi), i8 = j8/(n/pi); // j5, j6, i7, i8 are used to calculate weight index
					if(j8%(n/pi)>=ki*ki*hi*wi*ti || i8+pi*i9>=co || ki*ki*i7+ki*(j5%ki)+j6%ki>=ci || (j6/ki)-(fw-1)/2+i2 < 0 || (j6/ki)-(fw-1)/2+i2 > wi-1 || (j5/ki)-(fh-1)/2+i1 < 0 || (j5/ki)-(fh-1)/2+i1 > hi-1)
						// compact_weight_vec[i1][i2][i9][j8] = 0.0;// 建议改成continue
						continue;	// wfl change
					else
					{
						compact_weight_vec[i1][i2][i9][j8] = weight[i1][i2][ki*ki*i7+ki*(j5%ki)+j6%ki][i8+pi*i9];
					}
				}
			}
		}
	}

	cout<<"select one setting..."<<endl;
	pFile<<"select one setting..."<<endl;
	// select one setting
	for(int j4=0; j4<co; j4++)
	{
		for(int v1=0; v1<ko*ho; v1++)
		{
			for(int v2=0; v2<ko*wo; v2++)
			{
				for(int u3=0; u3<to; u3++)
				{
					// if(ko*ko*u3 + ko*(v1%ko) + v2%ko == j4)	select_one[j4][v1][v2][u3] = constant_weight[j4] / sqrt(running_var[j4]+epsilon);
					// else select_one[j4][v1][v2][u3] = 0.0;// 没有必要在赋值了
					if(ko*ko*u3 + ko*(v1%ko) + v2%ko != j4) continue;										// wfl change
					else select_one[j4][v1][v2][u3] = constant_weight[j4] / sqrt(running_var[j4]+epsilon);	// wfl change
				}
			}
		}
	}

	// select one vector setting
	for(int j4=0; j4<co; j4++)
	{
		for(int v1=0; v1<ko*ho; v1++)
		{
			for(int v2=0; v2<ko*wo; v2++)
			{
				for(int u3=0; u3<to; u3++)
				{
					select_one_vec[j4][ko*ko*ho*wo*u3 + ko*wo*v1 + v2] = select_one[j4][v1][v2][u3];
				}
			}
		}
	}

	// ciphertext variables
	Ciphertext *ctxt_in=&cipher_pool[0], *ct_zero=&cipher_pool[1], *temp=&cipher_pool[2], *sum=&cipher_pool[3], *total_sum=&cipher_pool[4], *var=&cipher_pool[5];

	// ciphertext input
	*ctxt_in = cnn_in.cipher();

	// rotated input precomputation
	vector<vector<Ciphertext*>> ctxt_rot(fh, vector<Ciphertext*>(fw));
	cout<< "fw: " <<fw << "\tfh: " <<fh<<endl;pFile<< "fw: " <<fw << "\tfh: " <<fh <<endl;
	// if(fh != 3 || fw != 3) throw std::invalid_argument("fh and fw should be 3");
	if(fh%2 == 0 || fw%2 == 0) throw std::invalid_argument("fh and fw should be odd");
	cout<<"rotated input precomputation start\n";pFile<<"rotated input precomputation start\n";
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			if(i1==(fh-1)/2 && i2==(fw-1)/2) ctxt_rot[i1][i2] = ctxt_in;		// i1=(fh-1)/2, i2=(fw-1)/2 means ctxt_in
			else if((i1==(fh-1)/2 && i2>(fw-1)/2) || i1>(fh-1)/2) ctxt_rot[i1][i2] = &cipher_pool[6+fw*i1+i2-1];
			else ctxt_rot[i1][i2] = &cipher_pool[6+fw*i1+i2];
		}
	}
	cout<<"rotated input precomputation end!!!"<<endl;pFile<<"rotated input precomputation end!!!"<<endl;
	// ctxt_rot[0][0] = &cipher_pool[6];	ctxt_rot[0][1] = &cipher_pool[7];	ctxt_rot[0][2] = &cipher_pool[8];	
	// ctxt_rot[1][0] = &cipher_pool[9];	ctxt_rot[1][1] = ctxt_in;			ctxt_rot[1][2] = &cipher_pool[10];		// i1=1, i2=1 means ctxt_in
	// ctxt_rot[2][0] = &cipher_pool[11];	ctxt_rot[2][1] = &cipher_pool[12];	ctxt_rot[2][2] = &cipher_pool[13];
	// 密文位置转换
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			*ctxt_rot[i1][i2] = *ctxt_in;
			memory_save_rotate(*ctxt_rot[i1][i2], *ctxt_rot[i1][i2], ki*ki*wi*(i1-(fh-1)/2) + ki*(i2-(fw-1)/2), evaluator, gal_keys);
		}
	}

	// generate zero ciphertext 
	cout<<"create zero ciphertext start!\n";
	pFile<<"create zero ciphertext start!\n";
	vector<double> zero(1<<logn, 0.0);
	Plaintext plain;
	encoder.encode(zero, ctxt_in->scale(), plain);
	encryptor.encrypt(plain, *ct_zero);		// ct_zero: original scaling factor
	cout<<"create zero ciphertext end!\n";
	pFile<<"create zero ciphertext end!\n";

	for(int i9=0; i9<q; i9++)
	{
		// weight multiplication
		cout << "multiplication by filter coefficients..." << endl;
		pFile << "multiplication by filter coefficients..." << endl;
		for(int i1=0; i1<fh; i1++)
		{
			for(int i2=0; i2<fw; i2++)
			{
				// *temp = *ctxt_in;
				// memory_save_rotate(*temp, *temp, k*k*l*(i1-(kernel-1)/2) + k*(i2-(kernel-1)/2), scale_evaluator, gal_keys);
				// scale_evaluator.multiply_vector_inplace_scaleinv(*temp, compact_weight_vec[i1][i2][i9]);		// temp: double scaling factor
				evaluator.multiply_vector_reduced_error(*ctxt_rot[i1][i2], compact_weight_vec[i1][i2][i9], *temp);		// temp: double scaling factor
				if(i1==0 && i2==0) *sum = *temp;	// sum: double scaling factor
				else evaluator.add_inplace_reduced_error(*sum, *temp);
			}
		}
		evaluator.rescale_to_next_inplace(*sum);
		*var = *sum;

		// summation for all input channels   (wfl：归并相加会不会结果好一点)
		//---------------可尝试优化start----------------
		cout << "summation for all input channels..." << endl;
		pFile << "summation for all input channels..." << endl;
		int d = log2_long(ki), c = log2_long(ti);
		cout << "多路复用并行卷积之中b步骤的矩阵大小 d = log2_long(ki) = " << d << endl;
		pFile <<"多路复用并行卷积之中b步骤的矩阵大小 d = log2_long(ki) = " << d << endl;
		cout << "多路复用并行卷积之中b步骤的矩阵大小 c = log2_long(ti) = " << c << endl;
		pFile <<"多路复用并行卷积之中b步骤的矩阵大小 c = log2_long(ti) = " << c << endl;
		for(int x=0; x<d; x++)
		{
			*temp = *var;
		//	scale_evaluator.rotate_vector(temp, pow2(x), gal_keys, temp);
			memory_save_rotate(*temp, *temp, pow2(x), evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(*var, *temp);
		}
		for(int x=0; x<d; x++)
		{
			*temp = *var;
		//	scale_evaluator.rotate_vector(temp, pow2(x)*k*l, gal_keys, temp);
			memory_save_rotate(*temp, *temp, pow2(x)*ki*wi, evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(*var, *temp);
		}
		if(c==-1)
		{
			cout << "c = -1 时进入，此时ti： " << ti << endl;
			pFile << "c = -1 时进入，此时ti： " << ti << endl;
			*sum = *ct_zero;
			for(int x=0; x<ti; x++)
			{
				*temp = *var;
			//	scale_evaluator.rotate_vector(temp, k*k*l*l*x, gal_keys, temp);
				memory_save_rotate(*temp, *temp, ki*ki*hi*wi*x, evaluator, gal_keys);
				evaluator.add_inplace_reduced_error(*sum, *temp);
			}
			*var = *sum;
		}
		else
		{
			for(int x=0; x<c; x++)
			{
				*temp = *var;
			//	scale_evaluator.rotate_vector(temp, pow2(x)*k*k*l*l, gal_keys, temp);
				memory_save_rotate(*temp, *temp, pow2(x)*ki*ki*hi*wi, evaluator, gal_keys);
				evaluator.add_inplace_reduced_error(*var, *temp);
			}
		}
		// 最终数据保存在var中，temp用于中间计算，sum用于中间计算
		//---------------优化end----------------

		// collecting valid values into one ciphertext.
		cout << "collecting valid values into one ciphertext..." << endl;
		pFile << "collecting valid values into one ciphertext..." << endl;
		for(int i8=0; i8<pi && pi*i9+i8<co; i8++)
		{
			int j4 = pi*i9+i8;
			if(j4 >= co) throw std::out_of_range("the value of j4 is out of range!");

			*temp = *var;
			memory_save_rotate(*temp, *temp, (n/pi)*(j4%pi) - j4%ko - (j4/(ko*ko))*ko*ko*ho*wo - ((j4%(ko*ko))/ko)*ko*wo, evaluator, gal_keys);
			evaluator.multiply_vector_inplace_reduced_error(*temp, select_one_vec[j4]);		// temp: double scaling factor
			if(i8==0 && i9==0) *total_sum = *temp;	// total_sum: double scaling factor
			else evaluator.add_inplace_reduced_error(*total_sum, *temp);
		}
	}
	evaluator.rescale_to_next_inplace(*total_sum);
	*var = *total_sum;

	// po copies
	// ？？？？if中目的是什么？？？？
	if(end == false)
	{
		cout << "po copies" << endl;pFile << "po copies" << endl;
		*sum = *ct_zero;
		for(int u6=0; u6<po; u6++)
		{
			*temp = *var;
			memory_save_rotate(*temp, *temp, -u6*(n/po), evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(*sum, *temp);		// sum: original scaling factor.
		}
		*var = *sum;
	}

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, *var);
	cout<<"multiplexed parallel convolution over!!!\n";pFile<<"multiplexed parallel convolution over!!!\n";

}
void multiplexed_parallel_batch_norm_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> bias, vector<double> running_mean, vector<double> running_var, vector<double> weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, double B, bool end)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = ki, ho = hi, wo = wi, co = ci, to = ti, po = pi;

	// error check
	if(static_cast<int>(bias.size())!=ci || static_cast<int>(running_mean.size())!=ci || static_cast<int>(running_var.size())!=ci || static_cast<int>(weight.size())!=ci) throw std::invalid_argument("the size of bias, running_mean, running_var, or weight are not correct");
	for(auto num : running_var) if(num<pow(10,-16) && num>-pow(10,-16)) throw std::invalid_argument("the size of running_var is too small. nearly zero.");
	if(hi*wi*ci > 1<<logn) throw std::invalid_argument("hi*wi*ci should not be larger than n");

	// generate g vector
	vector<double> g(1<<logn, 0.0);

	// set f value
	long n = 1<<logn;

	// check if pi | n
	if(n%pi != 0) throw std::out_of_range("n is not divisible by pi");

	cout<<"g是用来计算y=ax+b的 vector: ";
	pFile<<"g是用来计算y=ax+b的 vector: ";
	// set g vector
	for(int v4=0; v4<n; v4++)
	{
		int v1 = ((v4%(n/pi))%(ki*ki*hi*wi))/(ki*wi), v2 = (v4%(n/pi))%(ki*wi), u3 = (v4%(n/pi))/(ki*ki*hi*wi);
		if(ki*ki*u3+ki*(v1%ki)+v2%ki>=ci || v4%(n/pi)>=ki*ki*hi*wi*ti) g[v4] = 0.0;
		else 
		{
			int idx = ki*ki*u3 + ki*(v1%ki) + v2%ki;
			g[v4] = (running_mean[idx] * weight[idx] / sqrt(running_var[idx]+epsilon) - bias[idx])/B;
		}
		// cout<<g[v4]<<" ";pFile<<g[v4]<<" ";
	}
	// cout<<endl;pFile<<endl;

	// encode & encrypt
	Plaintext plain;
	Ciphertext cipher_g;
	Ciphertext temp;
	temp = cnn_in.cipher();
	encoder.encode(g, temp.scale(), plain);
	encryptor.encrypt(plain, cipher_g);

	// batch norm
	evaluator.sub_inplace_reduced_error(temp, cipher_g);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, temp);
	cout<<"multiplexed parallel batch norm end!\n";
	pFile<<"multiplexed parallel batch norm end!\n";

}
void ReLU_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, long alpha, vector<Tree> &tree, double scaled_val, long scalingfactor, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double scale)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = ki, ho = hi, wo = wi, co = ci, to = ti, po = pi;

	// error check
	if(hi*wi*ci > 1<<logn) throw std::invalid_argument("hi*wi*ci should not be larger than n");

	// ReLU
	Ciphertext temp;
	temp = cnn_in.cipher();
	minimax_ReLU_seal(comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, temp, temp);
	// 激活函数换成别的公式(待使用)
	// evaluator.square_inplace(temp);
	// evaluator.relinearize_inplace(temp, relin_keys); 
    // evaluator.rescale_to_next_inplace(temp);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, temp);
}

void adv_ReLU_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, long alpha, vector<Tree> &tree, double scaled_val, long scalingfactor, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double scale)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = ki, ho = hi, wo = wi, co = ci, to = ti, po = pi;

	// error check
	if(hi*wi*ci > 1<<logn) throw std::invalid_argument("hi*wi*ci should not be larger than n");

	// ReLU
	Ciphertext temp;
	temp = cnn_in.cipher();
	// minimax_ReLU_seal(comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, temp, temp);
	// 激活函数换成别的公式(待使用)
	evaluator.square_inplace(temp);
	evaluator.relinearize_inplace(temp, relin_keys); 
    evaluator.rescale_to_next_inplace(temp);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, temp);
}

void adv_ReLU1_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, long alpha, vector<Tree> &tree, double scaled_val, long scalingfactor, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double scale)
{
	// 0.0593965793152903+0.4999999842286903*x+0.7958675094533638*x^2+(4.751539162128893e-9)*x^3-0.370170930053213*x^4
	cout<<"0.0593965793152903+0.4999999842286903*x+0.7958675094533638*x^2+2(4.751539162128893e-9)*x^3-0.370170930053213*x^4"<<endl;
	pFile<<"0.0593965793152903+0.4999999842286903*x+0.7958675094533638*x^2+2(4.751539162128893e-9)*x^3-0.370170930053213*x^4"<<endl;
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = ki, ho = hi, wo = wi, co = ci, to = ti, po = pi;

	// error check
	if(hi*wi*ci > 1<<logn) throw std::invalid_argument("hi*wi*ci should not be larger than n");

	// ReLU
	Ciphertext temp1,temp2,temp3,temp4,ct1,ct2,ct3,ct4,ct5;
	temp1 = cnn_in.cipher();
	temp2 = cnn_in.cipher();
	temp3 = cnn_in.cipher();
	temp4 = cnn_in.cipher();
	Plaintext plain1,plain2,plain3,plain4,plain5;
	encoder.encode(-1.72785485941069, temp1.scale(), plain1);
	evaluator.mod_switch_to_inplace(plain1, temp1.parms_id());
	encryptor.encrypt(plain1, ct1);
	encoder.encode(0.15813344207366, temp2.scale(), plain2);
	evaluator.mod_switch_to_inplace(plain2, temp2.parms_id());
	encryptor.encrypt(plain2, ct2);
	encoder.encode(0.61530420931439, temp3.scale(), plain3);
	evaluator.mod_switch_to_inplace(plain3, temp3.parms_id());
	encryptor.encrypt(plain3, ct3);
	encoder.encode(0.95441719518657, temp4.scale(), plain4);
	evaluator.mod_switch_to_inplace(plain4, temp4.parms_id());
	encryptor.encrypt(plain4, ct4);
	

	evaluator.add_inplace_reduced_error(temp1, ct1);
	evaluator.add_inplace_reduced_error(temp2, ct2);
	evaluator.add_inplace_reduced_error(temp3, ct3);
	evaluator.add_inplace_reduced_error(temp4, ct4);

	evaluator.multiply_inplace_reduced_error(temp1, temp2, relin_keys);
	evaluator.rescale_to_next_inplace(temp1);

	evaluator.multiply_inplace_reduced_error(temp3, temp4, relin_keys);
	evaluator.rescale_to_next_inplace(temp3);
	cout << "激活第一层乘法scale: temp1和temp3：" << temp1.scale() << " " << temp3.scale() << endl;
	pFile << "激活第一层乘法scale: temp1和temp3：" << temp1.scale() << " " << temp3.scale() << endl;

	evaluator.multiply_inplace_reduced_error(temp1, temp3, relin_keys);
	evaluator.rescale_to_next_inplace(temp1);
	cout << "激活第二层乘法scale: temp1：" << temp1.scale() << endl;
	pFile << "激活第二层乘法scale: temp1：" << temp1.scale() << endl;

	encoder.encode(-0.370170930053213, temp1.scale(), plain5);
	evaluator.mod_switch_to_inplace(plain5, temp1.parms_id());

	evaluator.multiply_plain_inplace(temp1, plain5);
    evaluator.rescale_to_next_inplace(temp1);
	cout << "激活第二层乘法之后的明文相乘scale: temp1：" << temp1.scale() << endl;
	pFile << "激活第二层乘法之后的明文相乘scale: temp1：" << temp1.scale() << endl;

	// minimax_ReLU_seal(comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, temp, temp);
	// 激活函数换成别的公式(待使用)
	// evaluator.square_inplace(temp);
	// evaluator.relinearize_inplace(temp, relin_keys); 
    // evaluator.rescale_to_next_inplace(temp);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, temp1);
}


void cnn_add_seal(const TensorCipher &cnn1, const TensorCipher &cnn2, TensorCipher &destination, Evaluator &evaluator)
{
	// parameter setting
	int k1 = cnn1.k(), h1 = cnn1.h(), w1 = cnn1.w(), c1 = cnn1.c(), t1 = cnn1.t(), p1 = cnn1.p(), logn1 = cnn1.logn();
	int k2 = cnn2.k(), h2 = cnn2.h(), w2 = cnn2.w(), c2 = cnn2.c(), t2 = cnn2.t(), p2 = cnn2.p(), logn2 = cnn2.logn();

	// error check
	if(k1!=k2 || h1!=h2 || w1!=w2 || c1!=c2 || t1!=t2 || p1!=p2 || logn1!=logn2) throw std::invalid_argument("the parameters of cnn1 and cnn2 are not the same");

	// addition
	Ciphertext temp1, temp2;
	temp1 = cnn1.cipher();
	temp2 = cnn2.cipher();
	evaluator.add_inplace_reduced_error(temp1, temp2);

	destination = TensorCipher(logn1, k1, h1, w1, c1, t1, p1, temp1);
}
// 下采样，降低特征的维度
void multiplexed_parallel_downsampling_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 0, ho = 0, wo = 0, co = 0, to = 0, po = 0;

	// parameter setting
	long n = 1<<logn;
	ko = 2*ki;
	ho = hi/2;
	wo = wi/2;
	to = ti/2;
	co = 2*ci;
	po = pow2(floor_to_int(log(static_cast<double>(n)/static_cast<double>(ko*ko*ho*wo*to)) / log(2.0)));
	std::cout<<"multiplexed parallel downsampling... "<<std::endl;
	pFile<<"multiplexed parallel downsampling... "<<std::endl;
	std::cout<<"ko: "<<ko<<"\tho: "<<ho<<"\two: "<<wo<<"\tco: "<<co<<"\tto: "<<to<<"\tpo: "<<po<<std::endl;
	pFile<<"ko: "<<ko<<"\tho: "<<ho<<"\two: "<<wo<<"\tco: "<<co<<"\tto: "<<to<<"\tpo: "<<po<<std::endl;

	// error check
	if(ti%8 != 0) throw std::invalid_argument("ti is not multiple of 8");
	if(hi%2 != 0) throw std::invalid_argument("hi is not even");
	if(wi%2 != 0) throw std::invalid_argument("wi is not even");
	if(n%po != 0) throw std::out_of_range("n is not divisible by po");		// check if po | n

	// variables
	vector<vector<vector<double>>> select_one_vec(ki, vector<vector<double>>(ti, vector<double>(1<<logn, 0.0)));
	Ciphertext ct, sum, temp;
	ct = cnn_in.cipher();

	// selecting tensor vector setting
	cout<<"selecting tensor vector setting..."<<endl;
	pFile<<"selecting tensor vector setting..."<<endl;
	for(int w1=0; w1<ki; w1++)
	{
		for(int w2=0; w2<ti; w2++)
		{
			for(int v4=0; v4<1<<logn; v4++)
			{
				int j5 = (v4%(ki*ki*hi*wi))/(ki*wi), j6 = v4%(ki*wi), i7 = v4/(ki*ki*hi*wi);
				if(v4<ki*ki*hi*wi*ti && (j5/ki)%2 == 0 && (j6/ki)%2 == 0 && (j5%ki) == w1 && i7 == w2) select_one_vec[w1][w2][v4] = 1.0;
				else select_one_vec[w1][w2][v4] = 0.0;
				std::cout<<select_one_vec[w1][w2][v4]<<" ";pFile<<select_one_vec[w1][w2][v4]<<" ";
			}
			std::cout<<std::endl;pFile<<std::endl;
		}
		std::cout<<"w1: "<<w1<<std::endl;pFile<<"w1: "<<w1<<std::endl;
	}

	for(int w1=0; w1<ki; w1++)
	{
		for(int w2=0; w2<ti; w2++)
		{
			temp = ct;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one_vec[w1][w2]);

			int w3 = ((ki*w2+w1)%(2*ko))/2, w4 = (ki*w2+w1)%2, w5 = (ki*w2+w1)/(2*ko);
			memory_save_rotate(temp, temp, ki*ki*hi*wi*w2 + ki*wi*w1 - ko*ko*ho*wo*w5 - ko*wo*w3 - ki*w4 - ko*ko*ho*wo*(ti/8), evaluator, gal_keys);// 为什么要旋转：因为要选择对应的元素
			if(w1==0 && w2==0) sum = temp;
			else evaluator.add_inplace_reduced_error(sum, temp);
			
		}
	}
	evaluator.rescale_to_next_inplace(sum);		// added
	ct = sum;

	// for fprime packing
	sum = ct;
	for(int u6=1; u6<po; u6++)
	{
		temp = ct;
		memory_save_rotate(temp, temp, -(n/po)*u6, evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(sum, temp);
	}
	ct = sum;
	cout << "multiplexed parallel downsampling end" << endl;
	pFile << "multiplexed parallel downsampling end" << endl;
	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, ct);

}
// 平均池化
void averagepooling_seal_scale(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, double B, CKKSEncoder &encoder, Decryptor &decryptor, ofstream &output)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 1, ho = 1, wo = 1, co = ci, to = ti;

	if(log2_long(hi) == -1) throw std::invalid_argument("hi is not power of two");
	if(log2_long(wi) == -1) throw std::invalid_argument("wi is not power of two");

	Ciphertext ct, temp, sum;
	ct = cnn_in.cipher();

	// sum_hiwi
	cout << "平均池化sum hiwi" << endl;
	pFile << "平均池化sum hiwi" << endl;
	// 将ct中的值相加，放在ct中
	// 将对应通道的对应位置的元素相加
	for(int x=0; x<log2_long(wi); x++)
	{
		temp = ct;
	//	scale_evaluator.rotate_vector_inplace(temp, pow2(x)*k, gal_keys);
		memory_save_rotate(temp, temp, pow2(x)*ki, evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(ct, temp);	// 将ct与temp相加，并保证是正确的
	}
	for(int x=0; x<log2_long(hi); x++)
	{
		temp = ct;
	//	scale_evaluator.rotate_vector_inplace(temp, pow2(x)*k*k*l, gal_keys);
		memory_save_rotate(temp, temp, pow2(x)*ki*ki*wi, evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(ct, temp);
	}
	// 1  2  3  4 		 1  2  4
	// 5  6  7  8   ---> 5  6  8
	// 9  10 11 12  ---> 13 14 16
	// 13 14 15 16		设矩阵为4X4，池化后为2X2，选择对应元素为上面位置

	// cout << "sum l^2 results" << endl;
	// output << "sum l^2 results" << endl;
	// pFile << "sum l^2 results" << endl;
	// decrypt_and_print_txt(ct, decryptor, encoder, 1<<logn, 256, 2, output);
	// decrypt_and_print(ct, decryptor, encoder, 1<<logn, 256, 2);

	// final
	// cout << "final" << endl;
	// pFile << "final" << endl;
	vector<double> select_one(1<<logn, 0.0), zero(1<<logn, 0.0);
	for(int s=0; s<ki; s++)
	{
		for(int u=0; u<ti; u++)
		{
			int p=ki*u+s;
			temp = ct;
		//	scale_evaluator.rotate_vector_inplace(temp, -p*k + k*k*l*l*u + k*l*s, gal_keys);
			memory_save_rotate(temp, temp, -p*ki + ki*ki*hi*wi*u + ki*wi*s, evaluator, gal_keys);
			select_one = zero;
			// for(int i=0; i<k; i++) select_one[(k*u+s)*k+i] = 1.0 / static_cast<double>(l*l);
			for(int i=0; i<ki; i++) select_one[(ki*u+s)*ki+i] = B / static_cast<double>(hi*wi);
			
			std::cout<<" select one vector:\n";
			for(int i=0;i<select_one.size();++i){
				std::cout<<select_one[i]<<" ";
			}
			std::cout<<"\n";
			
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one);
			if(u==0 && s==0) sum = temp;	// double scaling factor
			else evaluator.add_inplace_reduced_error(sum, temp);
		}

		// cout << "final iteration results" << endl;
		// output << "final iteration results" << endl;
		// pFile << "final iteration results" << endl;
		// decrypt_and_print_txt(sum, decryptor, encoder, 1<<logn, 256, 2, output);
	}
	evaluator.rescale_to_next_inplace(sum);

	// cout << "rescaling results" << endl;
	// output << "rescaling results" << endl;
	// pFile << "rescaling results" << endl;
	// decrypt_and_print_txt(sum, decryptor, encoder, 1<<logn, 256, 2, output);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, 1, sum);
	
}
// 矩阵乘法 matrix乘以cnn_in，用于全链接层
void matrix_multiplication_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> matrix, vector<double> bias, int q, int r, Evaluator &evaluator, GaloisKeys &gal_keys)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = ki, ho = hi, wo = wi, co = ci, to = ti, po = pi;

	if(static_cast<int>(matrix.size()) != q*r) throw std::invalid_argument("the size of matrix is not q*r");
	if(static_cast<int>(bias.size()) != q) throw std::invalid_argument("the size of bias is not q");

	// generate matrix and bias
	vector<vector<double>> W(q+r-1, vector<double>(1<<logn, 0.0));
	vector<double> b(1<<logn, 0.0);

	for(int z=0; z<q; z++) b[z] = bias[z];
	for(int i=0; i<q; i++)
	{
		for(int j=0; j<r; j++)
		{
			W[i-j+r-1][i] = matrix[i*r+j];
			if(i-j+r-1<0 || i-j+r-1>=q+r-1) throw std::out_of_range("i-j+r-1 is out of range");
			if(i*r+j<0 || i*r+j>=static_cast<int>(matrix.size())) throw std::out_of_range("i*r+j is out of range");
		}
	}

	// matrix multiplication
	Ciphertext ct, temp, sum;
	ct = cnn_in.cipher();
	for(int s=0; s<q+r-1; s++)
	{
		temp = ct;
	//	scale_evaluator.rotate_vector_inplace(temp, r-1-s, gal_keys);
		memory_save_rotate(temp, temp, r-1-s, evaluator, gal_keys);
		evaluator.multiply_vector_inplace_reduced_error(temp, W[s]);// mod_switch_to_inplace + multiply_plain_inplace，明文乘以密文
		if(s==0) sum = temp;
		else evaluator.add_inplace_reduced_error(sum, temp);	// 将ct与temp相加，并保证是正确的
	}
	evaluator.rescale_to_next_inplace(sum);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, sum);

}

// 函数作用：保存旋转后的密文
void memory_save_rotate(const Ciphertext &cipher_in, Ciphertext &cipher_out, int steps, Evaluator &evaluator, GaloisKeys &gal_keys)
{
	long n = cipher_in.poly_modulus_degree() / 2;
	Ciphertext temp = cipher_in;
	steps = (steps+n)%n;	// 0 ~ n-1
	int first_step = 0;

	if(34<=steps && steps<=55) first_step = 33;
	else if(57<=steps && steps<=61) first_step = 33;
	else first_step = 0;
	if(steps == 0) return;		// no rotation

	if(first_step == 0) evaluator.rotate_vector_inplace(temp, steps, gal_keys);
	else
	{
		evaluator.rotate_vector_inplace(temp, first_step, gal_keys);	// ??
		evaluator.rotate_vector_inplace(temp, steps-first_step, gal_keys);
	}

	cipher_out = temp;
//	else scale_evaluator.rotate_vector(cipher_in, steps, gal_keys, cipher_out);
}

