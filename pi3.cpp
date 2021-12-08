//computing pi with FFT and Chudnovsky
#ifdef _MSC_VER
#pragma warning(disable:4996)   //  fopen() 
#endif
#include <stdc++.h>
#include <algorithm>
#include <memory>
#include <pmmintrin.h>
using namespace std;
namespace FFT_Pi {    
    void dump_to_file(const char* path, const std::string& str) {
        //  ghi vào file dưới dạng chuỗi string
        FILE* file = fopen(path, "wb");
        if (file == NULL)
            throw "Cannot Create File";
        fwrite(str.c_str(), 1, str.size(), file);
        fclose(file);
    }
    //  Fast Fourier Transform
#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif
    struct SIMD_delete {
        /*Loại dữ liệu __m128d, để sử dụng với nội dung hướng dẫn của
          phần mở rộng SIMD 2 của Truyền trực tuyến,
          được định nghĩa trong <pmmintrin.h>.
          _m128d:vectơ của hai số thực*/
        void operator()(__m128d* p) {
            _mm_free(p);
        }
    };
    struct my_complex {
        double r;
        double i;
        my_complex(double _r, double _i) : r(_r), i(_i) {};
    };
    std::vector<std::vector<my_complex>> twiddle_table;
    void fft_ensure_table(int k) {
        //  Đảm bảo bảng hệ số twiddle đủ lớn để xử lý FFT có kích thước 2 ^ k.
        int current_k = (int)twiddle_table.size() - 1;
        if (current_k >= k)  return;
        // Tính từng cấp độ một
        if (k - 1 > current_k)  fft_ensure_table(k - 1);
        size_t length = (size_t)1 << k;
        double omega = 2 * M_PI / length;
        length /= 2;
        //  Tạo sub-table.
        std::vector<my_complex> sub_table;
        for (size_t c = 0; c < length; c++) {
            //  Tạo các hệ số Twiddle 
            double angle = omega * c;
            auto twiddle_factor = my_complex(cos(angle), sin(angle));
            sub_table.push_back(twiddle_factor);
        }
        //  Đẩy vào table chính.
        twiddle_table.push_back(std::move(sub_table));
    }
    void fft_forward(__m128d* T, int k) {
        //  Fast Fourier Transform
        //  Hàm này thực hiện FFT chuyển tiếp có độ dài 2 ^ k.
        //  Đây là FFT phân rã theo tần số (DIF).
        //Các thông số:
        //  -   T           -   con trỏ tới mảng.
        //  -   k           -   2^k là kích thước của phép biến đổi

        //  Kết thúc đệ quy tại 2 điểm.
        if (k == 1) {
            __m128d a = T[0];
            __m128d b = T[1];
            T[0] = _mm_add_pd(a, b);// trả về vector tổng của a và b  [ a[0]+b[0], a[1]+b[1] ]
            T[1] = _mm_sub_pd(a, b);// trả về vector hiệu của a và b  [ a[0]-b[0], a[1]-b[1] ]
            return;
        }
        size_t length = (size_t)1 << k;
        size_t half_length = length / 2;
        //  Nhập local twiddle table.
        std::vector<my_complex>& local_table = twiddle_table[k];
        //  Đề quy FFT thành 2 nửa
        for (size_t c = 0; c < half_length; c++) {
            //  Lấy hệ số của Twiddle 
            __m128d r0 = _mm_loaddup_pd(&local_table[c].r); //sao chép
            __m128d i0 = _mm_loaddup_pd(&local_table[c].i);
            //  Lấy các phần tử của nó
            __m128d a0 = T[c];
            __m128d b0 = T[c + half_length];
            //  Perform butterfly
            __m128d c0, d0;
            c0 = _mm_add_pd(a0, b0);
            d0 = _mm_sub_pd(a0, b0);
            T[c] = c0;
            // Nhân với hệ số twiddle.
            c0 = _mm_mul_pd(d0, r0);
            d0 = _mm_mul_pd(_mm_shuffle_pd(d0, d0, 1), i0);
            c0 = _mm_addsub_pd(c0, d0); //	 Trừ và cộng
            T[c + half_length] = c0;
        }
        // Thực hiện đệ quy FFT trên các phần tử dưới.
        fft_forward(T, k - 1);
        // Thực hiện đệ quy FFT trên các phần tử trên.
        fft_forward(T + half_length, k - 1);
    }
    void fft_inverse(__m128d* T, int k) {
        //  Fast Fourier Transform
        //  Hàm này thực hiện FFT nghịch đảo có độ dài 2 ^ k.
         // Đầu vào phải theo thứ tự đảo ngược bit.
        //Các thông số:
        //  -   T           -   con trỏ tới mảng.
        //  -   k           -   2^k là kích thước của phép biến đổi

        // Kết thúc đệ quy tại 2 điểm.
        if (k == 1) {
            __m128d a = T[0];
            __m128d b = T[1];
            T[0] = _mm_add_pd(a, b);
            T[1] = _mm_sub_pd(a, b);
            return;
        }
        size_t length = (size_t)1 << k;
        size_t half_length = length / 2;
        // Thực hiện đệ quy FFT trên các phần tử dưới.
        fft_inverse(T, k - 1);
        //  Thực hiện đệ quy FFT trên các phần tử trên.
        fft_inverse(T + half_length, k - 1);
        //  Nhập local twiddle table.
        std::vector<my_complex>& local_table = twiddle_table[k];
        //  Thực hiện đệ quy FFT thành hai nửa.
        for (size_t c = 0; c < half_length; c++) {
            //  Lấy hệ số của Twiddle 
            __m128d r0 = _mm_loaddup_pd(&local_table[c].r);
            __m128d i0 = _mm_loaddup_pd(&local_table[c].i);
            i0 = _mm_xor_pd(i0, _mm_set1_pd(-0.0));
            //  Lấy các phần tử
            __m128d a0 = T[c];
            __m128d b0 = T[c + half_length];
            //  Perform butterfly
            __m128d c0, d0;
            //   Nhân với hệ số twiddle
            c0 = _mm_mul_pd(b0, r0);
            d0 = _mm_mul_pd(_mm_shuffle_pd(b0, b0, 1), i0);
            c0 = _mm_addsub_pd(c0, d0);
            b0 = _mm_add_pd(a0, c0);
            d0 = _mm_sub_pd(a0, c0);
            T[c] = b0;
            T[c + half_length] = d0;
        }
    }
    void fft_pointwise(__m128d* T, __m128d* A, int k) {
        // Thực hiện phép nhân của hai mảng FFT.
        size_t length = (size_t)1 << k; //2^k
        for (size_t c = 0; c < length; c++) {
            __m128d a0 = T[c];
            __m128d b0 = A[c];
            __m128d c0, d0;
            c0 = _mm_mul_pd(a0, _mm_unpacklo_pd(b0, b0));
            d0 = _mm_mul_pd(_mm_shuffle_pd(a0, a0, 1), _mm_unpackhi_pd(b0, b0));
            //shuffle [a0.r, a0.i]
            T[c] = _mm_addsub_pd(c0, d0);
            //   [((c0.r *d0.i)+(c0.i*d0.r)),
            //  ((c0.r*d0.r)-(c0.i*d0.i))]
        }
    }  
    void int_to_fft(__m128d* T, int k, const uint32_t* A, size_t AL) {
        //  Chuyển mảng word thành mảng FFT. Đặt 3 chữ số thập phân trên mỗi điểm của số phức.

        //:Các thông số
        //  -   T   -   FFT array
        //  -   k   -   2^k là kích thước của phép biến đổi
        //  -   A   -   mảng word
        //  -   AL  -   kích thước của mảng word

        size_t fft_length = (size_t)1 << k;
        __m128d* Tstop = T + fft_length;
        // Vì mỗi word có 9 chữ số và chúng ta muốn đặt 3 chữ số mỗi từ
         // điểm, độ dài của biến đổi ít nhất phải gấp 3 lần từ
         // độ dài của đầu vào.
        if (fft_length < 3 * AL)
            throw "FFT length is too small.";
        //  Convert
        for (size_t c = 0; c < AL; c++) {
            uint32_t word = A[c];
            *T++ = _mm_set_sd(word % 1000);
            word /= 1000;
            *T++ = _mm_set_sd(word % 1000);
            word /= 1000;
            *T++ = _mm_set_sd(word);
        }
        //  Đánh dấu phần còn lại bằng các số không
        while (T < Tstop)
            *T++ = _mm_setzero_pd();
    }
    void fft_to_int(__m128d* T, int k, uint32_t* A, size_t AL) {
        //  Chuyển đổi mảng FFT trở lại mảng word. Thực hiện quy tròn và lấy ra.
        //Thông số:
        //  -   T   -   mảng FFT 
        //  -   A   -   mảng word 
        //  -   AL  -   kích thước mảng word 

        //  Tính hệ số tỉ lệ
        size_t fft_length = (size_t)1 << k;
        double scale = 1. / fft_length;
       // Vì mỗi word có 9 chữ số và chúng ta muốn đặt 3 chữ số mỗi từ
         // điểm, độ dài của biến đổi ít nhất phải gấp 3 lần từ
         // độ dài của đầu vào.
        if (fft_length < 3 * AL)
            throw "FFT length is too small.";
        //  Làm tròn và tính tiếp.
        uint64_t carry = 0;
        for (size_t c = 0; c < AL; c++) {
            double   f_point;
            uint64_t i_point;
            uint32_t word;
            f_point = ((double*)T++)[0] * scale;    //  Load và scale
            i_point = (uint64_t)(f_point + 0.5);    //  làm tròn
            carry += i_point;                       //  + thêm vào carry
            word = carry % 1000;                    //  lấy 3 chữ số
            carry /= 1000;

            f_point = ((double*)T++)[0] * scale;    
            i_point = (uint64_t)(f_point + 0.5);    
            carry += i_point;                       
            word += (carry % 1000) * 1000;          
            carry /= 1000;

            f_point = ((double*)T++)[0] * scale;    
            i_point = (uint64_t)(f_point + 0.5);    
            carry += i_point;                       
            word += (carry % 1000) * 1000000;       
            carry /= 1000;

            A[c] = word;
        }
    }
    //  BigFloat object
    /*  Đây là object dấu phẩy động với kích thước lớn. Nó thể hiện một độ chính xác tùy ý số điểm 
     *
     *  Giá trị số của nó bằng:
     *
     *      word = 10^9
     *      word^exp * (T[0] + T[1]*word + T[2]*word^2 + ... + T[L - 1]*word^(L - 1))
     *
     *  T là một mảng các số nguyên 32 bit. Mỗi số nguyên lưu trữ 9 chữ số thập phân
     * và phải luôn có giá trị trong phạm vi [0, 999999999].
     *
     * T [L - 1] không bao giờ được bằng không.
     *
     * Số dương khi (sign = true) và âm khi (sign = false).
     * Số không được biểu diễn dưới dạng (sign = true) và (L = 0).   
     */
#define YCL_BIGFLOAT_EXTRA_PRECISION    2
    class BigFloat {
    public:
        BigFloat(BigFloat&& x);
        BigFloat& operator=(BigFloat&& x);

        BigFloat();
        BigFloat(uint32_t x, bool sign = true);

        std::string to_string(size_t digits = 0) const;
        std::string to_string_sci(size_t digits = 0) const;
        size_t get_precision() const;
        int64_t get_exponent() const;
        uint32_t word_at(int64_t mag) const;

        void negate();
        BigFloat mul(uint32_t x) const;
        BigFloat add(const BigFloat& x, size_t p = 0) const;
        BigFloat sub(const BigFloat& x, size_t p = 0) const;
        BigFloat mul(const BigFloat& x, size_t p = 0) const;
        BigFloat rcp(size_t p) const;
        BigFloat div(const BigFloat& x, size_t p) const;
    private:
        bool sign;      //  true mang giá trị dương hoặc 0 , false mang giá trị âm
        int64_t exp;    //  số mũ
        size_t L;       //  kích thước
        std::unique_ptr<uint32_t[]> T;
        //cấp phát động một đối tượng của unit_32t và trao quyền sở hữu đối tượng đó cho T
        //(tức là biến T sẽ chứa con trỏ trỏ tới vùng nhớ của đối tượng này)
        //  Internal helpers
        int64_t to_string_trimmed(size_t digits, std::string& str) const;
        int ucmp(const BigFloat& x) const;
        BigFloat uadd(const BigFloat& x, size_t p) const;
        BigFloat usub(const BigFloat& x, size_t p) const;
        friend BigFloat invsqrt(uint32_t x, size_t p);
    };
    BigFloat invsqrt(uint32_t x, size_t p);
    //  Move operators
    BigFloat::BigFloat(BigFloat&& x): sign(x.sign), exp(x.exp), L(x.L) , T(std::move(x.T))
    {
        x.sign = true;
        x.exp = 0;
        x.L = 0;
    }
    BigFloat& BigFloat::operator=(BigFloat&& x) {
        sign = x.sign;
        exp = x.exp;
        L = x.L;
        T = std::move(x.T);
        x.sign = true;
        x.exp = 0;
        x.L = 0;
        return *this;
    }
    //  Constructors
    BigFloat::BigFloat() : sign(true), exp(0) , L(0){}
    BigFloat::BigFloat(uint32_t x, bool sign_): sign(true), exp(0), L(1){
        //  Construct 1 BigFloat với 1 giá trị x là sign được chỉ định
        if (x == 0) {
            L = 0;  return;
        }
        sign = sign_;
        T = std::unique_ptr<uint32_t[]>(new uint32_t[1]);
        T[0] = x;
    }
    //  Chuyển đổi chuỗi string
    int64_t BigFloat::to_string_trimmed(size_t digits, std::string& str) const {
        //  Chuyển đổi đối tượng này thành một chuỗi có các số có nghĩa "chữ số".
        // Sau khi gọi hàm này, biểu thức sau bằng
        // giá trị số của object. (sau khi cắt bớt độ chính xác)
        // str "* 10 ^" (giá trị trả về)
        if (L == 0) {
            str = "0";
            return 0;
        }
        //  Thu thập toán hạng
        int64_t exponent = exp;
        size_t length = L;
        uint32_t* ptr = T.get();

        if (digits == 0)  digits = length * 9; // Sử dụng tất cả các chữ số.              
        else {
            // Cắt bớt độ chính xác
            size_t words = (digits + 17) / 9;
            if (words < length) {
                size_t chop = length - words;
                exponent += chop;
                length = words;
                ptr += chop;
            }
        }
        exponent *= 9;
        //  Xây dựng chuỗi string
        char buffer[] = "012345678";
        str.clear();
        size_t c = length;
        while (c-- > 0) {
            uint32_t word = ptr[c];
            for (int i = 8; i >= 0; i--) {
                buffer[i] = word % 10 + '0';
                word /= 10;
            }
            str += buffer;
        }
        //  Đếm số không ở đầu
        size_t leading_zeros = 0;
        while (str[leading_zeros] == '0')
            leading_zeros++;
        digits += leading_zeros;
        //  Cắt bớt
        if (digits < str.size()) {
            exponent += str.size() - digits;
            str.resize(digits);
        }
        return exponent;
    }
    std::string BigFloat::to_string(size_t digits) const {
        //  Chuyển số này thành chuỗi string. Tự động chọn kiểu dữ liệu.
        if (L == 0)    return "0.";
        int64_t mag = exp + L;
        // Sử dụng các kí hiệu rác nằm ngoài phạm vi
        if (mag > 1 || mag < 0)   return to_string_sci();
        //  Convert
        std::string str;
        int64_t exponent = to_string_trimmed(digits, str);
        //  Nếu ít hơn 1 
        if (mag == 0) {
            if (sign)  return std::string("0.") + str;
            else       return std::string("-0.") + str;
        }
        //  Tạo một chuỗi có các chữ số trước chữ số thập phân.
        std::string before_decimal = std::to_string((long long)T[L - 1]);
        //  Không có gì sau vị trí thập phân.
        if (exponent >= 0) {
            if (sign)   return before_decimal + ".";  
            else        return std::string("-") + before_decimal + ".";      
        }
        // Nhận các chữ số sau chữ số thập phân.
        std::string after_decimal = str.substr((size_t)(str.size() + exponent), (size_t)-exponent);
        if (sign)      return before_decimal + "." + after_decimal; 
        else           return std::string("-") + before_decimal + "." + after_decimal;     
    }
    std::string BigFloat::to_string_sci(size_t digits) const {
        // Chuyển thành chuỗi ký hiệu toán học
        if (L == 0)    return "0.";
        //  Chuyển đổi
        std::string str;
        int64_t exponent = to_string_trimmed(digits, str);
        // Dải các số không ở đầu.
            size_t leading_zeros = 0;
            while (str[leading_zeros] == '0')     leading_zeros++;
            str = &str[leading_zeros];
        //  Chèn vị trí thập phân
        exponent += str.size() - 1;
        str = str.substr(0, 1) + "." + &str[1];
        //  Thêm số mũ
        if (exponent != 0) {
            str += " * 10^";
            str += std::to_string(exponent);
        }
        //  Thêm dấu
        if (!sign)    str = std::string("-") + str;
        return str;
    }
    //  Getters
    size_t BigFloat::get_precision() const {
        // Trả về độ chính xác của một số trong các từ.
        // Mỗi word có 9 chữ số thập phân.
        return L;
    }
    int64_t BigFloat::get_exponent() const {
        // Trả về số mũ của một số trong các word.
        // Mỗi word có 9 chữ số thập phân.
        return exp;
    }
    uint32_t BigFloat::word_at(int64_t mag) const {
        // Trả về từ ở vị trí chữ số thứ nhất.
        // Điều này hữu ích cho các bổ sung cần truy cập vào một "vị trí chữ số" cụ thể
        // của toán hạng mà không cần phải lo lắng nếu nó nằm ngoài giới hạn.

        // Hàm này về mặt toán học :
        //      (return value) = floor(this * (10^9)^-mag) % 10^9
        if (mag < exp)       return 0;
        if (mag >= exp + (int64_t)L)         return 0;
        return T[(size_t)(mag - exp)];
    }
    int BigFloat::ucmp(const BigFloat& x) const {
        // Hàm so sánh bỏ qua dấu.
        // Điều này là cần thiết để xác định các phép trừ sẽ đi theo hướng nào.
        // Độ lớn
        int64_t magA = exp + L;
        int64_t magB = x.exp + x.L;
        if (magA > magB)        return 1;
        if (magA < magB)       return -1;
        //  So sánh
        int64_t mag = magA;
        while (mag >= exp || mag >= x.exp) {
            uint32_t wordA = word_at(mag);
            uint32_t wordB = x.word_at(mag);
            if (wordA < wordB)        return -1;
            if (wordA > wordB)        return 1;
            mag--;
        }
        return 0;
    }
    //  Arithmetic
    void BigFloat::negate() {
        //  Phủ định số
        if (L == 0)     return;
        sign = !sign;
    }
    BigFloat BigFloat::mul(uint32_t x) const {
        //  Nhân với một số nguyên không dấu 32 bit.
        if (L == 0 || x == 0)        return BigFloat();
        //  Tính toán các trường cơ bản.
        BigFloat z;
        z.sign = sign;
        z.exp = exp;
        z.L = L;
        //  Phân bổ phần định trị
        z.T = std::unique_ptr<uint32_t[]>(new uint32_t[z.L + 1]);
        uint64_t carry = 0;
        for (size_t c = 0; c < L; c++) {
            carry += (uint64_t)T[c] * x;                //  Nhân lên và + vào
            z.T[c] = (uint32_t)(carry % 1000000000);    //  Lưu trữ 9 chữ số dưới cùng
            carry /= 1000000000;                        //  Chuyển xuống 
        }
        //  Thực hiện tính toán
        if (carry != 0)         z.T[z.L++] = (uint32_t)carry;
        return z;
    }
    BigFloat BigFloat::uadd(const BigFloat& x, size_t p) const {
        // Thực hiện phép cộng bỏ qua dấu của hai toán hạng.
        // Độ lớn
        int64_t magA = exp + L;
        int64_t magB = x.exp + x.L;
        int64_t top = std::max(magA, magB);
        int64_t bot = std::min(exp, x.exp);
        // Độ dài đối tượng
        int64_t TL = top - bot;

        if (p == 0) 
            //  Giá trị mặc định. Không có đường nối.
            p = (size_t)TL;       
        else 
            //  Tăng độ chính xác
            p += YCL_BIGFLOAT_EXTRA_PRECISION;
        // Thực hiện cắt ngắn chính xác.
        if (TL > (int64_t)p) {
            bot = top - p;
            TL = p;
        }
        // Tính toán các trường cơ bản.
        BigFloat z;
        z.sign = sign;
        z.exp = bot;
        z.L = (uint32_t)TL;
        // Phân bổ phần định trị
        z.T = std::unique_ptr<uint32_t[]>(new uint32_t[z.L + 1]);
        //  Thêm vào
        uint32_t carry = 0;
        for (size_t c = 0; bot < top; bot++, c++) {
            uint32_t word = word_at(bot) + x.word_at(bot) + carry;
            carry = 0;
            if (word >= 1000000000) {
                word -= 1000000000;
                carry = 1;
            }
            z.T[c] = word;
        }
        //  Thực hiện tính toán
        if (carry != 0)   z.T[z.L++] = 1;
        return z;
    }
    BigFloat BigFloat::usub(const BigFloat& x, size_t p) const {
        // Thực hiện phép trừ bỏ qua dấu của hai toán hạng.
        // "this" phải lớn hơn hoặc bằng x. Nếu không, phép toán không định nghĩa được.

        // Độ lớn
        int64_t magA = exp + L;
        int64_t magB = x.exp + x.L;
        int64_t top = std::max(magA, magB);
        int64_t bot = std::min(exp, x.exp);
        
        //Cắt bớt độ chính xác
        int64_t TL = top - bot;
        
        if (p == 0)  p = (size_t)TL;
        //  Giá trị mặc định. Không có đường nối.
             
        else p += YCL_BIGFLOAT_EXTRA_PRECISION;
        // Tăng độ chính xác
            
        
        if (TL > (int64_t)p) {
            bot = top - p;
            TL = p;
        }
        // Tính toán các trường cơ bản.
        BigFloat z;
        z.sign = sign;
        z.exp = bot;
        z.L = (uint32_t)TL;
        // Phân bổ phần định trị
        z.T = std::unique_ptr<uint32_t[]>(new uint32_t[z.L]);
        // phép trừ
        int32_t carry = 0;
        for (size_t c = 0; bot < top; bot++, c++) {
            int32_t word = (int32_t)word_at(bot) - (int32_t)x.word_at(bot) - carry;
            carry = 0;
            if (word < 0) {
                word += 1000000000;
                carry = 1;
            }
            z.T[c] = word;
        }
        // Dải các số không ở đầu
        while (z.L > 0 && z.T[z.L - 1] == 0)     z.L--;
        if (z.L == 0) {
            z.exp = 0;
            z.sign = true;
            z.T.reset();
        }
        return z;
    }
    BigFloat BigFloat::add(const BigFloat& x, size_t p) const {
        //  phép cộng

        // Độ chính xác của đối tượng là p.
        // Nếu (p = 0), thì không có việc cắt bớt nào được thực hiện. Toàn bộ hoạt động được thực hiện
        // ở độ chính xác tối đa mà không mất dữ liệu.

         // Cùng dấu. Thực hiện phép cộng.
        if (sign == x.sign)         return uadd(x, p);
        //  this > x
        if (ucmp(x) > 0)            return usub(x, p);
        //  this < x
        return x.usub(*this, p);
    }
    BigFloat BigFloat::sub(const BigFloat& x, size_t p) const {
        // Phép trừ

        // Độ chính xác của mục tiêu là p.
        // Nếu (p = 0), thì không có việc cắt bớt nào được thực hiện. Toàn bộ hoạt động được thực hiện
        // ở độ chính xác tối đa mà không mất dữ liệu.

        // Dấu khác nhau. Thực hiện phép cộng.
        if (sign != x.sign)
            return uadd(x, p);
        //  this > x
        if (ucmp(x) > 0)
            return usub(x, p);
        //  this < x
        BigFloat z = x.usub(*this, p);
        z.negate();
        return z;
    }
    BigFloat BigFloat::mul(const BigFloat& x, size_t p) const {
        //  Phép nhân

        // Độ chính xác của mục tiêu là p.
        // Nếu (p = 0), thì không có việc cắt bớt nào được thực hiện. Toàn bộ hoạt động được thực hiện
        // ở độ chính xác tối đa mà không mất dữ liệu.

           // Một trong hai toán hạng bằng không.
        if (L == 0 || x.L == 0)         return BigFloat();

        if (p == 0) p = L + x.L;
            //  Giá trị mặc định. Không có đường nối.           
        else   p += YCL_BIGFLOAT_EXTRA_PRECISION;  // Tăng độ chính xác         
        // Thu thập các toán hạng.
        int64_t Aexp = exp;
        int64_t Bexp = x.exp;
        size_t AL = L;
        size_t BL = x.L;
        uint32_t* AT = T.get();
        uint32_t* BT = x.T.get();

        // Thực hiện cắt ngắn chính xác.
        if (AL > p) {
            size_t chop = AL - p;
            AL = p;
            Aexp += chop;
            AT += chop;
        }
        if (BL > p) {
            size_t chop = BL - p;
            BL = p;
            Bexp += chop;
            BT += chop;
        }
        // Tính toán các trường cơ bản.
        BigFloat z;
        z.sign = sign == z.sign;    //  Sign mang giá trị true nếu các sign = nhau
        z.exp = Aexp + Bexp;       //  Thêm số mũ.
        z.L = AL + BL;           //   Thêm độ dài cho bây giờ. Có thể cần phải sửa lại sau.
        // Phân bổ phần định trị
        z.T = std::unique_ptr<uint32_t[]>(new uint32_t[z.L]);

        // Thực hiện phép nhân.
         // Xác định kích thước FFT tối thiểu.
        int k = 0;
        size_t length = 1;
        while (length < 3 * z.L) {
            length <<= 1;
            k++;
        }
        // Thực hiện một phép chập bằng FFT.
        // 3 chữ số mỗi điểm đủ nhỏ để không gặp lỗi làm tròn số
        // cho đến khi kích thước biến đổi là 2 ^ 30.
        // Độ dài biến đổi là 2 ^ 29 cho phép kích thước kết quả tối đa là
        // 2 ^ 29 * 3 = 1,610,612,736 chữ số thập phân.
        if (k > 29)
            throw "FFT size limit exceeded.";
        // Phân bổ mảng FFT
        SIMD_delete deletor;
        auto Ta = std::unique_ptr<__m128d[], SIMD_delete>((__m128d*)_mm_malloc(length * sizeof(__m128d), 16), deletor);
        auto Tb = std::unique_ptr<__m128d[], SIMD_delete>((__m128d*)_mm_malloc(length * sizeof(__m128d), 16), deletor);
        // Đảm bảo bảng twiddle đủ lớn.
        fft_ensure_table(k);

        int_to_fft(Ta.get(), k, AT, AL);           //  Convert toán hạng thứ 1
        int_to_fft(Tb.get(), k, BT, BL);           //  Convert toán hạng thứ 2
        fft_forward(Ta.get(), k);                  //  Transform toán hạng thứ 1
        fft_forward(Tb.get(), k);                  //  Transform toán hạng thứ 2
        fft_pointwise(Ta.get(), Tb.get(), k);      //  Nhân các điểm FFT
        fft_inverse(Ta.get(), k);                  //  Biến đổi FFT nghịch đảo
        fft_to_int(Ta.get(), k, z.T.get(), z.L);   //  Trả lại về mảng word
        // Kiểm tra từ hàng đầu và độ dài chính xác.
        if (z.T[z.L - 1] == 0)        z.L--;
        return z;
    }
    BigFloat BigFloat::rcp(size_t p) const {
        // Tính toán nghịch đảo bằng phương pháp Newton.
        //  r1 = r0 - (r0 * x - 1) * r0
        if (L == 0)
            throw "Divide by Zero";
        // Thu thập toán hạng
        int64_t Aexp = exp;
        size_t AL = L;
        uint32_t* AT = T.get();
        // Kết thúc đệ quy. Tạo điểm bắt đầu.
        if (p == 0) {
            // Cắt bớt độ chính xác thành 3.
            p = 3;
            if (AL > p) {
                size_t chop = AL - p;
                AL = p;
                Aexp += chop;
                AT += chop;
            }
            //  Chuyển đổi số thành dấu phẩy động.
            double val = AT[0];
            if (AL >= 2)
                val += AT[1] * 1000000000.;
            if (AL >= 3)
                val += AT[2] * 1000000000000000000.;
            // Tính toán đối ứng.
            val = 1. / val;
            Aexp = -Aexp;
            //  Tỉ lệ
            while (val < 1000000000.) {
                val *= 1000000000.;
                Aexp--;
            }
            // Xây dựng lại 1 BigFloat.
            uint64_t val64 = (uint64_t)val;
            BigFloat out;
            out.sign = sign;
            out.T = std::unique_ptr<uint32_t[]>(new uint32_t[2]);
            out.T[0] = (uint32_t)(val64 % 1000000000);
            out.T[1] = (uint32_t)(val64 / 1000000000);
            out.L = 2;
            out.exp = Aexp;
            return out;
        }
        // Độ chính xác một nửa
        size_t s = p / 2 + 1;
        if (p == 1) s = 0;
        if (p == 2) s = 1;
        // Đệ quy với một nửa độ chính xác
        BigFloat T = rcp(s);
        //  r1 = r0 - (r0 * x - 1) * r0
        return T.sub(this->mul(T, p).sub(BigFloat(1), p).mul(T, p), p);
    }
    BigFloat BigFloat::div(const BigFloat& x, size_t p) const {
        //  Phép chia
        return this->mul(x.rcp(p), p);
    }
    BigFloat invsqrt(uint32_t x, size_t p) {
        //  Tính toán căn bậc hai nghịch đảo bằng phương pháp Newton.
        //            (  r0^2 * x - 1  )
        //  r1 = r0 - (----------------) * r0
        //            (       2        )
        if (x == 0)
            throw "Divide by Zero";
        //  Kết thúc đệ quy. Tạo điểm bắt đầu.
        if (p == 0) {
            double val = 1. / sqrt((double)x);
            int64_t exponent = 0;
            //  Tỉ lệ
            while (val < 1000000000.) {
                val *= 1000000000.;
                exponent--;
            }
            //  Xây dựng lại 1 BigFloat.
            uint64_t val64 = (uint64_t)val;
            BigFloat out;
            out.sign = true;
            out.T = std::unique_ptr<uint32_t[]>(new uint32_t[2]);
            out.T[0] = (uint32_t)(val64 % 1000000000);
            out.T[1] = (uint32_t)(val64 / 1000000000);
            out.L = 2;
            out.exp = exponent;
            return out;
        }
        //  1 nửa độ chính xác
        size_t s = p / 2 + 1;
        if (p == 1) s = 0;
        if (p == 2) s = 1;
        // Làm lại với độ chính xác một nửa
        BigFloat T = invsqrt(x, s);
        BigFloat temp = T.mul(T, p);     //  r0^2
        temp = temp.mul(x);                 //  r0^2 * x
        temp = temp.sub(BigFloat(1), p); //  r0^2 * x - 1
        temp = temp.mul(500000000);         //  (r0^2 * x - 1) / 2
        temp.exp--;
        temp = temp.mul(T, p);               //  (r0^2 * x - 1) / 2 * r0
        return T.sub(temp, p);               //  r0 - (r0^2 * x - 1) / 2 * r0
    }
    //  Pi
    void Pi_BSR(BigFloat& P, BigFloat& Q, BigFloat& R, uint32_t a, uint32_t b, size_t p) {
        // Đệ quy Tách nhị phân cho Công thức Chudnovsky.
        if (b - a == 1) {
            // Tính trực tiếp P (a, a + 1), Q (a, a + 1) và R (a, a + 1) 
            //  P = (13591409 + 545140134 b)(2b-1)(6b-5)(6b-1) (-1)^b
            P = BigFloat(b).mul(545140134);
            P = P.add(BigFloat(13591409));
            P = P.mul(2 * b - 1);
            P = P.mul(6 * b - 5);
            P = P.mul(6 * b - 1);
            if (b % 2 == 1)
                P.negate();
            //  Q = 10939058860032000 * b^3
            Q = BigFloat(b);
            Q = Q.mul(Q).mul(Q).mul(26726400).mul(409297880);
            //  R = (2b-1)(6b-5)(6b-1)
            R = BigFloat(2 * b - 1);
            R = R.mul(6 * b - 5);
            R = R.mul(6 * b - 1);
            return;
        }
        // Tính đệ quy P (a, b), Q (a, b ) và R (a, b) 
        // m là trung điểm của a và b
        uint32_t m = (a + b) / 2;
        BigFloat P0, Q0, R0, P1, Q1, R1;
        Pi_BSR(P0, Q0, R0, a, m, p);
        Pi_BSR(P1, Q1, R1, m, b, p);
        P = P0.mul(Q1, p).add(P1.mul(R0, p), p);
        Q = Q0.mul(Q1, p);
        R = R0.mul(R1, p);
    }
    void Pi(size_t digits) {
        // Số 3 đứng đầu không được tính.
        digits++;
        size_t p = (digits + 8) / 9;
        size_t terms = (size_t)(p * 0.6346230241342037371474889163921741077188431452678) + 1;
        //  Limit Exceeded
        if ((uint32_t)terms != terms)
            throw "Limit Exceeded";
        cout << "Computing Pi..." << endl;
        cout << "Algorithm: Chudnovsky Formula" << endl << endl;
        double time0 = clock();  
        //cout << "Summing Series... " << terms << " terms" << endl;
        BigFloat P, Q, R;
        Pi_BSR(P, Q, R, 0, (uint32_t)terms, p);
        P = Q.mul(13591409).add(P, p);
        Q = Q.mul(4270934400);
        //cout << "Division... " << endl;
        P = Q.div(P, p);
        //cout << "InvSqrt... " << endl;
        Q = invsqrt(10005, p); 
        //cout << "Final Multiply... " << endl;
        P = P.mul(Q, p);
        double _time =clock();
        //cout << "Time: " << time4 - time3 << endl;      
        cout << "Total Time = " << (_time - time0)/ CLOCKS_PER_SEC << endl << endl;
        cout << "the data has been saved to the file pi.txt\n\n";
        dump_to_file("pi.txt", P.to_string(digits));
    }
}
void check() {
    ifstream text1file("pi.txt");
    string s1; 
    stringstream ss1;
    ss1 << text1file.rdbuf(); 
    s1 = ss1.str(); 

    ifstream text2file("check.txt");
    string s2;
    stringstream ss2;
    ss2 << text2file.rdbuf();
    s2 = ss2.str();

    for (int i = 0; i <= s1.length(); i++)
        if (s1[i] != s2[i]) 
        { 
            break;
            cout << "the results are correct to the " << i - 2 << " rd decimal\n";
            return;
        }
    cout << "the results are correct to the " << s1.length() - 2 << " rd decimal\n";
    return;
}
void name();
int main() {
    name();
    cout << "Number of digits of pi to calculate?\n";
    size_t digits ;
    cin >> digits;
    FFT_Pi::Pi(digits);
    check();
    return 0;
}
void name() {
    cout << "          * * * * * * * * * * * * * * * * * * * * * * * * *\n"
         << "          *           DO AN LAP TRINH TINH TOAN           *\n"
         << "          *           De tai: Tinh so Pi voi FFT          *\n"
         << "          *      Giao vien huong dan:   Pham Minh Tuan    *\n"
         << "          *      Sinh vien thuc hien:  Nguyen Anh Phuong  *\n"
         << "          *                            Hoang Ha Nhi       *\n"
         << "          * * * * * * * * * * * * * * * * * * * * * * * * *\n\n";
}

