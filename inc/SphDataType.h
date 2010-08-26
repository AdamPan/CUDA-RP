//
// SphDataTypes
//
#ifndef _DATA_TYPES_H_
#define _DATA_TYPES_H_

#include <string>

class SphData
{
private:
protected:
public:
    int m_svType;
    int m_dType;
    int m_dims[3];
    float m_orig[3];
    float m_pitch[3];
    int m_step;
    float m_time;
    float* m_pData;

    SphData();
    virtual~ SphData();

    bool Allocate();
    bool Allocate(const float value);
    bool Allocate(const int dims[3]);
    bool Deallocate();
    bool LoadSph(const std::string fname);
    bool SaveSph(const std::string fname);
    bool Info();
    /*
    inline float& GetValue(const int i, const int j, const int k){
      return m_pData[ i + j * m_dims[0] + k * m_dims[0] * m_dims[1] ];}
    inline float* GetPointer(const int i, const int j, const int k){
      return &m_pData[ i + j * m_dims[0] + k * m_dims[0] * m_dims[1] ];}
    inline void SetValue(const int i, const int j, const int k, const float data){
      m_pData[ i + j * m_dims[0] + k * m_dims[0] * m_dims[1] ] = data;}
      */
    float& GetValue(const int i, const int j, const int k);
    float* GetPointer(const int i, const int j, const int k);
    void SetValue(const int i, const int j, const int k, const float data);
};

inline float& SphData::GetValue(const int i, const int j, const int k)
{
    return m_pData[ i + j * m_dims[0] + k * m_dims[0] * m_dims[1] ];
}

inline float* SphData::GetPointer(const int i, const int j, const int k)
{
    return &m_pData[ i + j * m_dims[0] + k * m_dims[0] * m_dims[1] ];
}

inline void SphData::SetValue(const int i, const int j, const int k, const float data)
{
    m_pData[ i + j * m_dims[0] + k * m_dims[0] * m_dims[1] ] = data;
}

class SphXData : public SphData
{
public:
    int m_gdims[3];
    int m_gista[3];
    int m_band;

    SphXData();
//    SphXData(SphData sph);
    virtual~ SphXData();

    bool ImportSph(const SphData org);

    bool LoadSphX(const std::string fname);
    bool SaveSphX(const std::string fname);
    bool Info();
};

#endif // _DATA_TYPES_H_
