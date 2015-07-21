#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <sstream>
#include <stdio.h>
#include <math.h>
#include <vector>

using namespace std;

string* split(string s, string c);
double string_to_double( const std::string& s );
struct particle {
  int type;
  double pos[3];
};

int main(int argc, char *argv[] )
{
    if(argc!=2){
        cout << "usage: ./rdf xmlfile" << endl;
        return 0;
    }

    /// file output
    FILE * myOfile;
    myOfile =fopen("rdf.csv", "w");

    vector<particle> Pos;

    int count[]={0,0,0,0,0,0,0,0,0,0,0,0};

    int bins=200; /// RDF-bins for 20nm (200 Angström), bin size: 1 A =0.1nm
    int *rdfAA = new int[bins];
    int *rdfAB = new int[bins];
    int *rdfBA = new int[bins];
    int *rdfBB = new int[bins];
    float *rdfAAtotal = new float[bins];
    float *rdfBBtotal = new float[bins];
    for(int i=0; i<bins; i++){
        rdfAA[i]=0;
        rdfAB[i]=0;
        rdfBA[i]=0;
        rdfBB[i]=0;
        rdfAAtotal[i]=0;
        rdfBBtotal[i]=0;
    }

    int frame=0;

    string line;
    ifstream myfile (argv[1]);

    if (myfile.is_open())
    {
        while ( myfile.good() )
        {
            getline (myfile,line);
            if(line.compare("</tplgy_coords>")==0){
            //if(split(line, " ")[0].compare("<tplgy_coords")==0){
                frame++;
                //if(frame%100==0){
                    cout << "frame: " << frame << endl;
                //}

                /// RDF CALC
                for(int i = 0 ; i<Pos.size(); i++){
                    for(int j = i+1 ; j<Pos.size(); j++){
                        double x=Pos[i].pos[0]-Pos[j].pos[0];
                        double y=Pos[i].pos[1]-Pos[j].pos[1];
                        double z=Pos[i].pos[2]-Pos[j].pos[2];
                        double d=sqrt(x*x+y*y+z*z);
                        if(d<200){
                            if(Pos[i].type==0){
                                if(Pos[j].type==0){
                                    rdfAA[(int)floor(d)]++;
                                }
                                if(Pos[j].type==1){
                                    rdfAB[(int)floor(d)]++;
                                }
                            }
                            if(Pos[i].type==1){
                                if(Pos[j].type==0){
                                    rdfBA[(int)floor(d)]++;
                                }
                                if(Pos[j].type==1){
                                    rdfBB[(int)floor(d)]++;
                                }
                            }
                            //cout <<"["<<Pos[i].type<< ","<<Pos[j].type<<"] "<< Pos[i].pos[0] << " " << Pos[j].pos[0] << ":" << x << " " << Pos[i].pos[1]  << " " << Pos[j].pos[1] << ":" << y << " " << Pos[i].pos[2] << " " << Pos[j].pos[2] << ":" << z << "  " <<d<< endl;
                        }/// fi dist
                    }/// j
                }/// i
                for(int i=0; i<bins; i++){
                    rdfAAtotal[i]+=(float)rdfAA[i]/count[0];
                    rdfBBtotal[i]+=(float)rdfBB[i]/count[1];
                    rdfAA[i]=0;
                    rdfAB[i]=0;
                    rdfBA[i]=0;
                    rdfBB[i]=0;
                }
                Pos.clear();
                for(int i=0; i<12; i++){
                    count[i]=0;
                }
            }
            /// read the atom lines
            else if(split(line, " ")[0].compare("<p")==0){
                particle x;
                //<p id="0" type="1" c="-31.09250630276887,-42.34489538041901,34.654154873058644"/>
                x.type=atoi(split(line,"\"")[3].c_str());

                string pos=split(line,"\"")[5];
                x.pos[0]=string_to_double(split(pos,",")[0].c_str())*10;
                x.pos[1]=string_to_double(split(pos,",")[1].c_str())*10;
                x.pos[2]=string_to_double(split(pos,",")[2].c_str())*10;

                Pos.push_back(x);
                count[x.type]++;
            }
        }
    }
    myfile.close();

    for(int i=0; i<bins; i++){
        rdfAAtotal[i]=rdfAAtotal[i]/frame;
        rdfBBtotal[i]=rdfBBtotal[i]/frame;
    }


    /// rdf file output
    cout << frame << "frames"<<endl;
    cout<< "A: "<< count[0] << " B: " << count[2] <<endl;
    fprintf(myOfile,"d  RDF(AA) RDF(BB)\n");
    for(int i=0; i<bins;i++){
        fprintf(myOfile,"%i, %f, %f\n", i   , rdfAAtotal[i], rdfBBtotal[i]);
    }
    fclose(myOfile);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

/// http://stackoverflow.com/questions/392981/how-can-i-convert-string-to-double-in-c
double string_to_double( const std::string& s )
{
  std::istringstream i(s);
  double x;
  if (!(i >> x))
    return 0;
  return x;
}


string* split(string s, string c){
    /// s - text
    /// c - searchpatern
    //cout << s << endl << c << endl;
    int count = 0;
    for(int i=0; i<s.length()-c.length()+1;i++){
        //cout <<i << " " << s.substr(i,c.length()) ;
        if(c.compare(s.substr(i,c.length()))==0){
            count++;
            //cout << "x" ;
        }
        //cout << endl;
    }
    string* result = new string[count+1];
    ///
    int x = 0;
    count=0;
     for(int i=0; i<s.length()-c.length()+1;i++){
        if(c.compare(s.substr(i,c.length()))==0){
            result[count]=s.substr(x,i-x);
            //cout << i << " " << s.substr(x,i-x) << endl;
            count++;
            x=i+c.length();
        }
    }
     //cout <<endl<< x << " "<< s[x]<< endl;
     //cout << s.length()-x-2 << endl;
     result[count]=s.substr(x,s.length()-x-2);

     for(int i=0; i<count; i++){
         //cout << result[i]<< endl;
     }
    return result;
}
