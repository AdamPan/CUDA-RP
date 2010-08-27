//
// SphDataTypes
//
#include "SphDataType.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

SphData::SphData() : m_pData(NULL)
{
    m_svType = 1; //scalar
    m_dType = 1; //float
    m_dims[0] = m_dims[1] = m_dims[2] = 0;
    m_orig[0] = m_orig[1] = m_orig[2] = 0.;
    m_pitch[0] = m_pitch[1] = m_pitch[2] = 0.;
    m_step = 0;
    m_time = 0.;
}

SphData::~SphData()
{
//  cout << "m_pData = " << m_pData << endl;
//  cout << "&m_pData = " << &m_pData << endl;
    if ( m_pData ) SphData::Deallocate();
}

bool SphData::Allocate()
{
    m_pData = new float[ m_dims[0] * m_dims[1] * m_dims[2] ];
    for ( int n = 0; n < m_dims[0] * m_dims[1] * m_dims[2]; n++ )
    {
        m_pData[n] = 0.;
    }
    return true;
}

bool SphData::Allocate(const float value)
{
#ifdef DEBUG
    try
    {
        m_pData = new float[ m_dims[0] * m_dims[1] * m_dims[2] ];
        cout << "Allocate memory : " << m_dims[0]*m_dims[1]*m_dims[2] * sizeof(float)
             << "Byte" << endl;
    }
    catch (bad_alloc)
    {
        cerr << "Allocation fail" << endl;
    }
#else
    m_pData = new float[ m_dims[0] * m_dims[1] * m_dims[2] ];
#endif
    for ( int n = 0; n < m_dims[0] * m_dims[1] * m_dims[2]; n++ )
    {
        m_pData[n] = value;
    }
    return true;
}

bool SphData::Allocate(const int dims[3])
{
    m_dims[0] = dims[0];
    m_dims[1] = dims[1];
    m_dims[2] = dims[2];
    SphData::Allocate();
//  m_pData = new float[ m_dims[0] * m_dims[1] * m_dims[2] ];
    return true;
}

bool SphData::Deallocate()
{
    if ( m_pData != NULL )
    {
        delete[] m_pData;
        m_pData = NULL;
        return true;
    }
    else return false;
}

bool SphData::LoadSph( const std::string fname )
{
    cout << "Load file name is " << fname << endl;

    ifstream is( fname.c_str(), ios::binary);

    if (! is.is_open() )
    {
        cout << "File could not be opened." << endl;
        return false;
    }

    int len;

    is.read((char*)&len, 4);
    is.read((char*)&m_svType, 4);
    is.read((char*)&m_dType, 4);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)m_dims, len);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)m_orig, len);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)m_pitch, len);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)&m_step, 4);
    is.read((char*)&m_time, 4);
    is.read((char*)&len, 4);

    SphData::Allocate( m_dims );

    is.read((char*)&len, 4);
    is.read((char*)m_pData, len);
    is.read((char*)&len, 4);

    is.close();

    if ( *(char*)&m_svType == 1 || *(char*)&m_svType == 2)
    {
        cout << "Little endian" << endl;
    }
    else
    {
        cout << "Big endian" << endl;
    }

    SphData::Info();

    return true;
}

bool SphData::SaveSph(const std::string fname)
{
#ifdef DEBUG
    cout << "Save file name is " << fname << endl;
#endif
    ofstream os( fname.c_str(), ios::binary);

    if (! os.is_open() ) return false;

    int len;
    len = 8;
    os.write((char*)&len, 4);
    os.write((char*)&m_svType, 4);
    os.write((char*)&m_dType, 4);
    os.write((char*)&len, 4);

    len = 12;
    os.write((char*)&len, 4);
    os.write((char*)m_dims, len);
    os.write((char*)&len, 4);

    len = 12;
    os.write((char*)&len, 4);
    os.write((char*)m_orig, len);
    os.write((char*)&len, 4);

    len = 12;
    os.write((char*)&len, 4);
    os.write((char*)m_pitch, len);
    os.write((char*)&len, 4);

    len = 8;
    os.write((char*)&len, 4);
    os.write((char*)&m_step, 4);
    os.write((char*)&m_time, 4);
    os.write((char*)&len, 4);

    len = 4 * m_dims[0] * m_dims[1] * m_dims[2];
    os.write((char*)&len, 4);
    os.write((char*)m_pData, len);
    os.write((char*)&len, 4);

    os.close();

#ifdef DEBUG
    	SphData::Info();
#endif
    return true;
}

bool SphData::Info()
{
    cout << "+++ Information of SPH Data +++" << endl;
    switch ( m_svType )
    {
    case 1:
        cout << "svType is scalar." << endl;
        break;
    case 2:
        cout << "svType is vector." << endl;
        break;
    }
    switch ( m_dType )
    {
    case 1:
        cout << "dType is float." << endl;
        break;
    case 2:
        cout << "dType is double." << endl;
        break;
    }
    cout << "imax = " << m_dims[0] << ", jmax = " << m_dims[1] << ", kmax = " << m_dims[2] << endl;
    cout << "xorig = " << m_orig[0] << ", yorig = " << m_orig[1] << ", zorig = " << m_orig[2] << endl;
    cout << "xpitch = " << m_pitch[0] << ", ypitch = " << m_pitch[1] << ", zpitch = " << m_pitch[2] << endl;
    cout << "step = " << m_step << ", time = " << m_time << endl;

    return true;
}

SphXData::SphXData()
{
    m_gdims[0] = m_gdims[1] = m_gdims[2] = 0;
    m_gista[0] = m_gista[1] = m_gista[2] = 1;
    m_band = 0;
}

SphXData::~SphXData()
{
}

bool SphXData::ImportSph(const SphData org)
{
    m_svType = org.m_svType;
    m_dType = org.m_dType;
    m_dims[0] = org.m_dims[0];
    m_dims[1] = org.m_dims[1];
    m_dims[2] = org.m_dims[2];
    m_orig[0] = org.m_orig[0];
    m_orig[1] = org.m_orig[1];
    m_orig[2] = org.m_orig[2];
    m_pitch[0] = org.m_pitch[0];
    m_pitch[1] = org.m_pitch[1];
    m_pitch[2] = org.m_pitch[2];
    m_step = org.m_step;
    m_time = org.m_time;
    m_pData = org.m_pData;
    m_gdims[0] = org.m_dims[0];
    m_gdims[1] = org.m_dims[1];
    m_gdims[2] = org.m_dims[2];
    m_gista[0] = m_gista[1] = m_gista[2] = 1;
    m_band = 0;
    return true;
}

bool SphXData::LoadSphX( const std::string fname )
{
    cout << "Load file name is " << fname << endl;

    ifstream is( fname.c_str(), ios::binary);

    if (! is.is_open() )
    {
        cout << "File could not be opened." << endl;
        return false;
    }

    int len;

    is.read((char*)&len, 4);
    is.read((char*)&m_svType, 4);
    is.read((char*)&m_dType, 4);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)m_dims, len);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)m_orig, len);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)m_pitch, len);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)&m_step, 4);
    is.read((char*)&m_time, 4);
    is.read((char*)&len, 4);

    SphData::Allocate( m_dims );

    is.read((char*)&len, 4);
    is.read((char*)m_pData, len);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)m_gdims, len);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)m_gista, len);
    is.read((char*)&len, 4);

    is.read((char*)&len, 4);
    is.read((char*)&m_band, len);
    is.read((char*)&len, 4);

    is.close();

    if ( *(char*)&m_svType == 1 || *(char*)&m_svType == 2)
    {
        cout << "Little endian" << endl;
    }
    else
    {
        cout << "Big endian" << endl;
    }

    SphXData::Info();

    return true;
}

bool SphXData::SaveSphX(const std::string fname)
{
    cout << "Save file name is " << fname << endl;

    ofstream os( fname.c_str(), ios::binary);

    if (! os.is_open() ) return false;

    int len;
    len = 8;
    os.write((char*)&len, 4);
    os.write((char*)&m_svType, 4);
    os.write((char*)&m_dType, 4);
    os.write((char*)&len, 4);

    len = 12;
    os.write((char*)&len, 4);
    os.write((char*)m_dims, len);
    os.write((char*)&len, 4);

    len = 12;
    os.write((char*)&len, 4);
    os.write((char*)m_orig, len);
    os.write((char*)&len, 4);

    len = 12;
    os.write((char*)&len, 4);
    os.write((char*)m_pitch, len);
    os.write((char*)&len, 4);

    len = 8;
    os.write((char*)&len, 4);
    os.write((char*)&m_step, 4);
    os.write((char*)&m_time, 4);
    os.write((char*)&len, 4);

    len = 4 * m_dims[0] * m_dims[1] * m_dims[2];
    os.write((char*)&len, 4);
    os.write((char*)m_pData, len);
    os.write((char*)&len, 4);

    len = 12;
    os.write((char*)&len, 4);
    os.write((char*)m_gdims, len);
    os.write((char*)&len, 4);

    len = 12;
    os.write((char*)&len, 4);
    os.write((char*)m_gista, len);
    os.write((char*)&len, 4);

    len = 4;
    os.write((char*)&len, 4);
    os.write((char*)&m_band, len);
    os.write((char*)&len, 4);

    os.close();

    SphXData::Info();

    return true;
}

bool SphXData::Info()
{
    SphData::Info();

//    switch ( m_svType )
//    {
//      case 1:
//        cout << "svType is scalar." << endl;
//        break;
//      case 2:
//        cout << "svType is vector." << endl;
//        break;
//    }
//    switch ( m_dType )
//    {
//      case 1:
//        cout << "dType is float." << endl;
//        break;
//      case 2:
//        cout << "dType is double." << endl;
//        break;
//    }
//      cout << "imax = " << m_dims[0] << ", jmax = " << m_dims[1] << ", kmax = " << m_dims[2] << endl;
//      cout << "xorig = " << m_orig[0] << ", yorig = " << m_orig[1] << ", zorig = " << m_orig[2] << endl;
//      cout << "xpitch = " << m_pitch[0] << ", ypitch = " << m_pitch[1] << ", zpitch = " << m_pitch[2] << endl;
//      cout << "step = " << m_step << ", time = " << m_time << endl;


    cout << "imax_global = " << m_gdims[0] << ", jmax_global = " << m_gdims[1] << ", kmax_global = " << m_gdims[2] << endl;
    cout << "ista = " << m_gista[0] << ", jsta = " << m_gista[1] << ", ksta = " << m_gista[2] << endl;
    cout << "band = " << m_band << endl;
    return true;
}
