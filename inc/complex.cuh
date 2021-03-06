struct __align__ (16) doublecomplex
{
    double real;
    double imag;
};

__inline__ __host__ __device__ doublecomplex make_doublecomplex (const double a, const double b)
{
    doublecomplex c;
    c.real = a, c.imag = b;
    return c;
}

__inline__ __host__ __device__ doublecomplex operator+ (const doublecomplex a, const doublecomplex b)
{
    return make_doublecomplex(a.real + b.real, a.imag + b.imag);
}

__inline__ __host__ __device__ doublecomplex operator+ (const double a, const doublecomplex b)
{
    return make_doublecomplex(a + b.real, b.imag);
}

__inline__ __host__ __device__ doublecomplex operator+ (const doublecomplex a, const double b)
{
    return make_doublecomplex(a.real + b, a.imag);
}

__inline__ __host__ __device__ doublecomplex operator- (const doublecomplex a, const doublecomplex b)
{
    return make_doublecomplex(a.real - b.real, a.imag - b.imag);
}

__inline__ __host__ __device__ doublecomplex operator- (const doublecomplex a, const double b)
{
    return make_doublecomplex(a.real - b, a.imag);
}

__inline__ __host__ __device__ doublecomplex operator- (const double a, const doublecomplex b)
{
    return make_doublecomplex(a - b.real, -b.imag);
}

__inline__ __host__ __device__ doublecomplex operator* (const doublecomplex a, const doublecomplex b)
{
    return make_doublecomplex(a.real * b.real - a.imag * b.imag, a.imag * b.real + a.real * b.imag);
}

__inline__ __host__ __device__ doublecomplex operator* (const doublecomplex a, const double b)
{
    return make_doublecomplex(a.real * b, a.imag * b);
}

__inline__ __host__ __device__ doublecomplex operator* (const double a, const doublecomplex b)
{
    return make_doublecomplex(a * b.real, a * b.imag);
}

__inline__ __host__ __device__ doublecomplex operator/ (const doublecomplex a, const doublecomplex b)
{
    double c = (b.real * b.real + b.imag * b.imag);
    return make_doublecomplex((a.real * b.real + a.imag * b.imag)/c, (a.imag * b.real - a.real * b.imag)/c);
}

__inline__ __host__ __device__ doublecomplex operator/ (const doublecomplex a, const double b)
{
    return make_doublecomplex(a.real / b, a.imag / b);
}

__inline__ __host__ __device__ doublecomplex operator/ (const double a, const doublecomplex b)
{
    double c = (b.real * b.real + b.imag * b.imag);
    return make_doublecomplex(a * b.real / c, -a * b.imag / c);
}

__inline__ __host__ __device__ double abs (const doublecomplex a)
{
    return sqrt(a.real * a.real + a.imag * a.imag);
}

__inline__ __host__ __device__ doublecomplex coth(doublecomplex a)
{
    double x = tanh(a.real), y = tan(a.imag);
    return make_doublecomplex(1.0, x * y)/make_doublecomplex(x,y);
}

__inline__ __host__ __device__ bool is_nan(doublecomplex a)
{
    return isnan(a.real) || isnan(a.imag);
}
