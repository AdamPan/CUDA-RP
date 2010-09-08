static __inline__ __host__ __device__ double2 operator* (const double2 a, const double2 b)
{
    return make_double2(a.x * b.x, a.y * b.y);
}

static __inline__ __host__ __device__ double2 operator* (const double2 a, const double b)
{
    return make_double2(a.x * b, a.y * b);
}

static __inline__ __host__ __device__ double2 operator* (const double a, const double2 b)
{
    return make_double2(a * b.x, a * b.y);
}

static __inline__ __host__ __device__ double2 operator+ (const double2 a, const double2 b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}

static __inline__ __host__ __device__ double2 operator+ (const double2 a, const double b)
{
    return make_double2(a.x + b, a.y + b);
}

static __inline__ __host__ __device__ double2 operator+ (const double a, const double2 b)
{
    return make_double2(a + b.x, a + b.y);
}

static __inline__ __host__ __device__ double2 operator+= (const double2 a, const double2 b)
{
    return a + b;
}

static __inline__ __host__ __device__ double2 operator- (const double2 a, const double2 b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}

static __inline__ __host__ __device__ double2 operator- (const double2 a, const double b)
{
    return make_double2(a.x - b, a.y - b);
}

static __inline__ __host__ __device__ double2 operator- (const double a, const double2 b)
{
    return make_double2(a - b.x, a - b.y);
}

static __inline__ __host__ __device__ double2 operator/ (const double2 a, const double2 b)
{
    return make_double2(a.x / b.x, a.y / b.y);
}

static __inline__ __host__ __device__ double2 operator/ (const double2 a, const double b)
{
    return make_double2(a.x / b, a.y / b);
}

static __inline__ __host__ __device__ double2 operator/ (const double a, const double2 b)
{
    return make_double2(a / b.x, a / b.y);
}

static __inline__ __host__ __device__ double sum(const double2 a)
{
    return a.x + a.y;
}
