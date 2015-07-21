 

#include <ReaDDyGPU.hpp>
#include <string>
#include <vector>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <cstdio>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>


// own string split
string* split(string s, string c){
    int x = 0;
    int count = 0;
    //<p id="0" type="1" c="-31.09250630276887,-42.34489538041901,34.654154873058644"/>
    string* result = new string[7];
    for(int i=0; i<s.length();i++){
        if(s[i]==c[0]){
            result[count]=s.substr(x,i-x);
            count++;
            x=i+c.length();
        }
    }
    result[count]=s.substr(x,s.length()-x);
    return result;
}

// string to double function
/// http://stackoverflow.com/questions/392981/how-can-i-convert-string-to-double-in-c
double string_to_double( const std::string& s )
{
  std::istringstream i(s);
  double x;
  if (!(i >> x))
    return 0;
  return x;
}

int CoordStringToDoubleArray(string in, double out[3]){
    in.erase(0,1);
    in.erase(in.length()-1,1);
    //cout << in << endl;
    for(int i=0; i<2; ++i){
        //cout << in << " " << in.find_first_of(',') << endl;
        //cout << "-- " << in.substr(0,in.find_first_of(',')).c_str() << endl;
        out[i]=atof(in.substr(0,in.find_first_of(',')).c_str());
        //cout << "-- " << in.substr(0,in.find_first_of(',')).c_str() << endl;
        in.erase(0, in.find_first_of(',')+1);
    }
    //cout << "-- " << in.substr(0,in.find_first_of(',')).c_str() << endl;
    out[2]=atof(in.substr(0,in.find_first_of(',')).c_str());
    return 1;
}

int * stringToIntArray(string inputString ){

    vector<int> myVector;
    inputString.erase(0,1);
    inputString.erase(inputString.length()-1,1);
    while(inputString.find_first_of(',')!=string::npos){
        myVector.push_back(atoi(inputString.substr(0,inputString.find_first_of(',')).c_str()) );
        inputString.erase(0, inputString.find_first_of(','));
    };
    myVector.push_back(atoi(inputString.c_str()));
    int * intArray = new int[myVector.size()];
    for(int j=0; j<myVector.size(); ++j){
        intArray[j] = myVector[j];
    }
    return intArray;
}

vector<int> stringToIntVector(string inputString ){

    vector<int> myVector;
    if(inputString.length()<=1)
        return myVector;
    /// delete [ and ] from the ends
    inputString.erase(0,1);
    inputString.erase(inputString.length()-1,1);
    while(inputString.find_first_of(',')!=string::npos){
        myVector.push_back(atoi(inputString.substr(0,inputString.find_first_of(',')).c_str()) );
        inputString.erase(0, inputString.find_first_of(','));
    };
    myVector.push_back(atof(inputString.c_str()));
    return myVector;
}

double * stringToDoubleArray(string inputString ){

    vector<double> myVector;
    inputString.erase(0,1);
    inputString.erase(inputString.length()-1,1);
    while(inputString.find_first_of(',')!=string::npos){
        myVector.push_back(atof(inputString.substr(0,inputString.find_first_of(',')).c_str()) );
        inputString.erase(0, inputString.find_first_of(','));
    };
    myVector.push_back(atof(inputString.c_str()));
    double * doubleArray = new double[myVector.size()];
    for(int j=0; j<myVector.size(); ++j){
        doubleArray[j] = myVector[j];
    }
    return doubleArray;
}

vector<int> stringToIntIntVector(string inputString){
    vector<int> intIntVector;
    // "[1,1];[0,1]"
    if(inputString.length()<=1)
        return intIntVector;
    //cout << inputString << endl;
    while(inputString.find_first_of(';')!=string::npos){
        string s = inputString.substr(1, inputString.find_first_of(';')-2);
        //cout << s << endl;
        intIntVector.push_back(atoi(s.substr(0, s.find_first_of(",")).c_str()));
        intIntVector.push_back(atoi(s.substr(s.find_first_of(",")+1, s.length()-s.find_first_of(",")).c_str()));
        inputString.erase(0,inputString.find_first_of(';')+1);
    };
    //cout << inputString << endl;
    inputString.erase(0,1);
    inputString.erase(inputString.length()-1,1);
    string s = inputString;
    intIntVector.push_back(atoi(s.substr(0, s.find_first_of(",")).c_str()));
    intIntVector.push_back(atoi(s.substr(s.find_first_of(",")+1, s.length()-s.find_first_of(",")).c_str()));
    // cout << s << endl;
    //cout << intIntVector[intIntVector.size()-2] << " " << intIntVector[intIntVector.size()-1] << endl;

    return intIntVector;
}

// clock function
double getTime(timeval start){
    timeval end;
    gettimeofday(&end, 0);
    double sec =(double)(end.tv_sec-start.tv_sec);
    double usec = (double)(end.tv_usec-start.tv_usec);
    return(sec+(0.000001*usec) );
}
