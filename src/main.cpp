#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include "gurobi_c++.h"
using namespace std;
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <sys/time.h>
#include <sys/resource.h>

//double prvals[] = {0.046, 0.0964, 0.0125, 0.0568, 0.0984, 0.0388, 0.0819, 0.0728, 0.0591, 0.0037, 0.0963, 0.0645, 0.0255, 0.0442, 0.0462, 0.0894, 0.0597, 0.0579, 0.0196, 0.0828, 0.0855, 0.0510, 0.0691, 0.0024, 0.0506, 0.0707, 0.0684, 0.0418, 0.0598, 0.0296, 0.0695, 0.0042, 0.0387, 0.0932, 0.0486, 0.0050, 0.0461};

//INFERENCE
struct LvrLearner
{
vector<double> trueclass;
vector<double> predclass;
vector<double> correctpred;
vector<double> a_trueclass;
vector<double> a_predclass;
vector<double> a_correctpred;
ofstream logfile;
ofstream tmpresultsfile;
string outdir;
string mlnwtsfile;
string tmpresultsdir;
int numtrigfeatures;
bool printresults;
int numtrigtypes;
int numargtypes;
vector<vector<int> > triggerconfusion;
vector<vector<int> > argconfusion;
double lrate;
double regrate;
int numiterations;
double trigcfac;
double jcfac;
double alpha1;
double alpha2;
vector<vector<vector<vector<double> > > > jointweights;
vector<vector<vector<vector<double> > > > cjointweights;
int numwords;
int numedges;
vector<vector<vector<vector<double> > > > outrefwts;
vector<vector<vector<int> > > refindex;
vector<int> refupdate;
bool includesoft1;
LvrLearner()
{
	numtrigtypes = 4;
	numargtypes = 3;
	tmpresultsdir = "tmpgres/";
	printresults = false;
	trueclass.resize(numtrigtypes);
	predclass.resize(numtrigtypes);
	correctpred.resize(numtrigtypes);
	a_trueclass.resize(numargtypes);
	a_predclass.resize(numargtypes);
	a_correctpred.resize(numargtypes);

	lrate = 0.01;
	regrate = 0.001;
	numiterations = 50;
	trigcfac = 0.6;
	jcfac = 1.25;
	triggerconfusion.clear();
	triggerconfusion.resize(12);
	for(int i=0;i<triggerconfusion.size();i++)
		triggerconfusion[i].resize(12);
	argconfusion.clear();
	argconfusion.resize(numargtypes);
	for(int i=0;i<argconfusion.size();i++)
		argconfusion[i].resize(numargtypes);
	numwords = 2615;
	numedges = 173;
	jointweights.resize(numwords);
	for(int i=0;i<jointweights.size();i++)
	{
		vector<vector<vector<double> > > tmp(numedges);
		for(int j=0;j<tmp.size();j++)
		{
			vector<vector<double> > tmp1(12);
			for(int k=0;k<tmp1.size();k++)
			{
				vector<double> tmp2(3);
				tmp1[k] = tmp2;
			}
			tmp[j] = tmp1;
		}
		jointweights[i] = tmp;
	}

	cjointweights.resize(numwords);
	for(int i=0;i<jointweights.size();i++)
	{
		vector<vector<vector<double> > > tmp(numedges);
		for(int j=0;j<tmp.size();j++)
		{
			vector<vector<double> > tmp1(4);
			for(int k=0;k<tmp1.size();k++)
			{
				vector<double> tmp2(3);
				tmp1[k] = tmp2;
			}
			tmp[j] = tmp1;
		}
		cjointweights[i] = tmp;
	}
	if(includesoft1)
	{
		outrefwts.resize(numwords);
		for(int i=0;i<outrefwts.size();i++)
		{
			vector<vector<vector<double> > > tmpc(numedges);
			for(int j=0;j<tmpc.size();j++)
			{
				vector<vector<double> > tmp1c(3);
				for(int k=0;k<tmp1c.size();k++)
				{
					vector<double> tmp2c(12);
					tmp1c[k] = tmp2c;
				}
				tmpc[j] = tmp1c;
			}
			outrefwts[i] = tmpc;
		}
		refindex.resize(numwords);
		for(int i=0;i<numwords;i++)
			refindex[i].resize(numedges);
	}
}

void resetconfusion()
{
	printconfusion();
	triggerconfusion.clear();
	triggerconfusion.resize(numtrigtypes);
	for(int i=0;i<triggerconfusion.size();i++)
		triggerconfusion[i].resize(numtrigtypes);
	argconfusion.clear();
	argconfusion.resize(numargtypes);
	for(int i=0;i<argconfusion.size();i++)
		argconfusion[i].resize(numargtypes);
}

void printconfusion()
{
	for(int i=0;i<triggerconfusion.size();i++)
	{
		for(int j=0;j<triggerconfusion[i].size();j++)
			cout<<triggerconfusion[i][j]<<" ";
		cout<<endl;
	}
	cout<<endl;
	for(int i=0;i<argconfusion.size();i++)
	{
		for(int j=0;j<argconfusion[i].size();j++)
			cout<<argconfusion[i][j]<<" ";
		cout<<endl;
	}
	cout<<endl;
}
~LvrLearner()
{
	
}

void tokenize(const string& str,
                      vector<string>& tokens,
                      char delim)
{
    stringstream ss(str);
    string item;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
}

void tointarr(string s,vector<int>& arr)
{
	vector<string> sarr;
	tokenize(s,sarr,':');
	for(int i=0;i<sarr.size();i++)
	{
		int num;
		stringstream st(sarr[i]);
		st >> num;
		arr.push_back(num);
	}
}

string itos(int i) {stringstream s; s << i; return s.str(); }

void ILPnoconstr(vector<vector<vector<double> > > weights, vector<vector<double> > classweights, vector<vector<int> > jindex,
	vector<vector<int> > trigrefs,vector<vector<int> >& vals)
{
    vals.resize(classweights.size());
    for(int i=0;i<classweights.size();i++){
       double mx = -1000;
       int mxid = 0;
       vals[i].resize(weights[i].size());
       for(int j=0;j<classweights[i].size();j++){
          if(j==0)
          {
              double r = (double)rand()/RAND_MAX;
              if(r < 0.1)
                 classweights[i][j] = 0.5*classweights[i][j];
          }
          if(classweights[i][j] > mx){
             mx = classweights[i][j];
             mxid = j;
         }      
       }
       vals[i][0] = mxid;
       for(int j=1;j<weights[i].size();j++){
          mxid = 0;
          mx = -1000;
          for(int k=0;k<weights[i][j].size();k++){
            if(weights[i][j][k] > mx){
                mx = weights[i][j][k];
                mxid = k;
            }
          }
         vals[i][j] = mxid;
       }
    }
    return;
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);
	GRBVar*** gvars = new GRBVar**[weights.size()];
	GRBVar** classvars = new GRBVar*[weights.size()];
	int typevars = 14;
	for(int i=0;i<weights.size();i++)
	{
		gvars[i] = new GRBVar*[weights[i].size()];
		for(int j=0;j<weights[i].size();j++)
		{
		    int ntypes = numtrigtypes;
		    if(j > 0)
			    ntypes = numargtypes;
			gvars[i][j] = new GRBVar[ntypes];
			for(int k=0;k<ntypes;k++)
			{
			    string s = "G_" + itos(i) + "_" + itos(j) + "_" + itos(k);
			    gvars[i][j][k] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, s);
			}
		}
		classvars[i] = new GRBVar[typevars];
		for(int j=0;j<typevars;j++)
		{
			classvars[i][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "H_"+itos(i)+"_"+itos(j));
		}
	}
	model.update();


	GRBLinExpr obj = 0;

	for(int i=0;i<weights.size();i++)
	{
		GRBLinExpr cexpr = 0;
		for(int j=0;j<typevars;j++)
		{
			cexpr += classvars[i][j];
		}
		model.addConstr(cexpr == 1.0,"C_"+itos(i));
		for(int j=1;j<weights[i].size();j++)
		{
		    int ntypes = numargtypes;
			GRBLinExpr expr = 0;
			for(int k=0;k<ntypes;k++)
			    expr += gvars[i][j][k];
			string s = "Sum_" + itos(i) + "_" + itos(j);
			model.addConstr(expr == 1.0, s);
		}
	}
    for(int i=0;i<weights.size();i++){
        for(int j=1;j<weights[i].size();j++){
             obj += weights[i][j][0]*gvars[i][j][0]+weights[i][j][1]*gvars[i][j][1]+weights[i][j][2]*gvars[i][j][2];
        }
    }
    for(int i=0;i<weights.size();i++){
                obj += classweights[i][0]*classvars[i][0]+classweights[i][1]*classvars[i][1]+classweights[i][2]*classvars[i][2]+
                        classweights[i][3]*classvars[i][3]+classweights[i][4]*classvars[i][4]+classweights[i][5]*classvars[i][5]+
                        classweights[i][6]*classvars[i][6]+classweights[i][7]*classvars[i][7]+classweights[i][8]*classvars[i][8]+
                        classweights[i][9]*classvars[i][9]+classweights[i][10]*classvars[i][10]+classweights[i][11]*classvars[i][11]+classweights[i][12]*classvars[i][12]+
						classweights[i][13]*classvars[i][13];
    }

	model.setObjective(obj, GRB_MAXIMIZE);
	model.getEnv().set(GRB_IntParam_OutputFlag,0);
	model.optimize();

	vals.resize(weights.size());
	for(int i=0;i<weights.size();i++)
	{
		vals[i].resize(weights[i].size());
		double mx = classvars[i][0].get(GRB_DoubleAttr_X);
		int mxid = 0;
		for(int j=1;j<14;j++)
		{
			if(classvars[i][j].get(GRB_DoubleAttr_X) > mx)
			{
				mx = classvars[i][j].get(GRB_DoubleAttr_X);
				mxid = j;
			}
		}
		vals[i][0] = mxid;
		for(int j=1;j<weights[i].size();j++)
		{
			double amx;
			int amxid;
			for(int k=0;k<numargtypes;k++)
			{
				if(k==0)
				{
					amxid = 0;
					amx = gvars[i][j][k].get(GRB_DoubleAttr_X);
				}
				else if(gvars[i][j][k].get(GRB_DoubleAttr_X) > amx)
				{
					amxid = k;
					amx = gvars[i][j][k].get(GRB_DoubleAttr_X);
				}
			}
			vals[i][j] = amxid;
		}

	}
   for(int i=0;i<weights.size();i++)
    {
        for(int j=0;j<weights[i].size();j++)
        {
	    delete[] gvars[i][j];
        }
	delete[] gvars[i];
    }
	delete[] gvars;
	for(int i=0;i<weights.size();i++)
		delete[] classvars[i];
	delete[] classvars;

}



void ILPnew(vector<vector<vector<double> > > weights, vector<vector<double> > classweights, vector<vector<int> > jindex,
	vector<vector<int> > trigrefs,vector<vector<int> >& vals)
{
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);
	GRBVar*** gvars = new GRBVar**[weights.size()];
	GRBVar** classvars = new GRBVar*[weights.size()];
	int typevars = 14;
	for(int i=0;i<weights.size();i++)
	{
		gvars[i] = new GRBVar*[weights[i].size()];
		for(int j=0;j<weights[i].size();j++)
		{
		    int ntypes = numtrigtypes;
		    if(j > 0)
			    ntypes = numargtypes;
			gvars[i][j] = new GRBVar[ntypes];
			for(int k=0;k<ntypes;k++)
			{
			    string s = "G_" + itos(i) + "_" + itos(j) + "_" + itos(k);
			    gvars[i][j][k] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, s);
			}
		}
		classvars[i] = new GRBVar[typevars];
		for(int j=0;j<typevars;j++)
		{
			classvars[i][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "H_"+itos(i)+"_"+itos(j));
		}
	}
	int numjointvars = 0;
	for(int i=0;i<jindex.size();i++)
	{
		if(jindex[i][0]==-1)
			continue;
		for(int j=1;j<jindex[i].size();j++)
		{
			if(jindex[i][j]==-1)
				continue;
			for(int k=0;k<12;k++)
			{
				for(int m=0;m<3;m++)
				{
					if(jointweights[jindex[i][0]][jindex[i][j]][k][m]==0)
						continue;
					numjointvars++;
				}
			}
			/*for(int k=0;k<4;k++)
			{
				for(int m=0;m<3;m++)
				{
					if(cjointweights[jindex[i][0]][jindex[i][j]][k][m]==0)
						continue;
					numjointvars++;
				}
			}*/
		}
	}
	GRBVar* jvars;
        //numjointvars = 0;
	if(numjointvars > 0)
	{
		jvars = new GRBVar [numjointvars];
		for(int i=0;i<numjointvars;i++)
		{
			jvars[i] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "jaux_"+itos(i));
		}
	}
	GRBVar* auxvars2 = new GRBVar [1];
	auxvars2[0] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "aux2_"+itos(0));

	GRBVar* refvars;
	if(includesoft1)
	{
		refvars = new GRBVar[100];
		for(int i=0;i<100;i++)
		{
			refvars[i] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "ref_"+itos(i));
		}
	}
	model.update();



	//Global limit on the number of triggers in the absence of regulation
	/*GRBLinExpr expreg = 0;
	for(int i=0;i<weights.size();i++)
	{
		expreg += gvars[i][0][2] + gvars[i][0][3];
	}
	model.addConstr(auxvars2[0] <= expreg,"SPL_REG_0");
	model.addConstr(expreg <= 100*auxvars2[0],"SPL_REG_1");
	GRBLinExpr lexp = 0;
	for(int i=0;i<weights.size();i++)
	{
		lexp += gvars[i][0][1];
	}
	model.addConstr(lexp - 1 - 100*auxvars2[0] <= 0,"SPL_REG3");
	*/
	GRBLinExpr obj = 0;
	if(includesoft1)
	{
		int refind = 0;
		for(int i=0;i<jindex.size();i++)
		{
			if(trigrefs[i].size()==0)
				continue;
			GRBLinExpr J1=0;
			int cx = 0;
			for(int j=0;j<trigrefs[i].size()-1;j+=2)
			{
				int tw = jindex[trigrefs[i][j]][0];
				int ad = jindex[trigrefs[i][j]][trigrefs[i][j+1]];
				if(tw==-1 || ad==-1)
					continue;
				for(int k=0;k<refindex[tw][ad].size();k+=2)
				{
					int a1 = refindex[tw][ad][k];
					int a2 = refindex[tw][ad][k+1];
					double v = outrefwts[tw][ad][a1][a2];
					int cval;
					if(a1==0)
						cval = 9;
					else if(a1==1)
						cval = 10;
					else
						cval = 11;
					if(refind < 100)
					{
					model.addConstr(2-classvars[i][a2]-classvars[trigrefs[i][j]][cval]+refvars[refind] >= 1,"JR_"+itos(i)+"_"+itos(j)+"_"+itos(k));
					model.addConstr(classvars[i][a2]+1-refvars[refind] >= 1,"JR1_"+itos(i)+"_"+itos(j)+"_"+itos(k));
					model.addConstr(classvars[trigrefs[i][j]][cval]+1-refvars[refind] >= 1,"JR2_"+itos(i)+"_"+itos(j)+"_"+itos(k));
					J1 += v*refvars[refind];
					cx++;
					refind++;
					}
				}
			}
			if(cx > 0)
			{
				obj += 0.001*J1*(1.0/(double)cx);
			}
		}
	}
	/*for(int i=0;i<jindex.size();i++)
	{
		if(trigrefs[i].size()==0)
			continue;
		for(int j=0;j<trigrefs[i].size()-1;j+=2)
		{
			if(trigrefs[trigrefs[i][j]].size()==0)
				continue;
			for(int k=0;k<trigrefs[trigrefs[i][j]].size();k+=2)
			{
				if(trigrefs[trigrefs[i][j]][k] == i)
				{
					int a1=trigrefs[trigrefs[i][j]][k];
					int a2=trigrefs[trigrefs[i][j]][k+1];
					//add a constraint to avoid cycles
					model.addConstr(gvars[trigrefs[i][j]][trigrefs[i][j+1]][0]+gvars[a1][a2][0] >=1,"CYC_"+itos(i)+"_"+itos(j)+"_"+itos(k));
					break;
				}
			}
		}
	}*/
	int auxind = 0;
    vector<vector<int> > isrefs(jindex.size());
    for(int i=0;i<jindex.size();i++)
    {
       isrefs[i].resize(jindex[i].size());
       for(int j=1;j<jindex[i].size();j++)
       {
          bool found=false;
          for(int k=0;k<jindex.size();k++)
          {
             if(k==i)
               continue;
             if(trigrefs[k].size()==0)
                 continue;
             for(int k1=0;k1<trigrefs[k].size();k1+=2)
             {
                 if(trigrefs[k][k1]==i && trigrefs[k][k1+1]==j)
                 {
                   found=true;
                   break;
                 }
             }
             if(found)
               break;
          }
          if(found)
            isrefs[i][j] = 1;
       }
    }
    if(numjointvars > 0){
	//joint soft
	for(int i=0;i<jindex.size();i++)
	{
		if(jindex[i][0]==-1)
			continue;
                int acnt = 0;
                GRBLinExpr jexp1 = 0;
                int validvals = 0;
		for(int j=1;j<jindex[i].size();j++)
		{
			if(jindex[i][j]==-1)
				continue;
                        validvals++;
                        GRBLinExpr J1 = 0;
                        acnt = 0;
                        double maxcwt = -1000;
                        int maxcwtind = 0;
                        for(int k=0;k<12;k++){
                           if(classweights[i][k] > maxcwt){
                               maxcwt = classweights[i][k];
                               maxcwtind = k;
                           }
                        }
			for(int k=0;k<12;k++)
			{
				for(int m=0;m<3;m++)
				{
					if(jointweights[jindex[i][0]][jindex[i][j]][k][m]==0)
						continue;
					model.addConstr(2-classvars[i][k]-gvars[i][j][m]+jvars[auxind] >= 1,"JL_"+itos(i)+"_"+itos(j)+"_"+itos(k)+"_"+itos(m));
					model.addConstr(classvars[i][k]+1-jvars[auxind] >= 1,"JL1_"+itos(i)+"_"+itos(j)+"_"+itos(k)+"_"+itos(m));
					model.addConstr(gvars[i][j][m]+1-jvars[auxind] >= 1,"JL2_"+itos(i)+"_"+itos(j)+"_"+itos(k)+"_"+itos(m));
				        //jexp1 += jvars[auxind]*jointweights[jindex[i][0]][jindex[i][j]][k][m];
                                        double conf = 1;
                                        if(k==7 && m==1 && classweights[i][k] > 0 && weights[i][j][m] > 0) 
                                           conf = 10;
                                        if(k==1 && m==1 && classweights[i][k] > 0 && weights[i][j][m] > 0){
                                             conf = 10;
                                        }
                                        if(k==3 && m==1 && classweights[i][k] > 0 && weights[i][j][m] > 0) 
                                           conf = 5;
                                        if(k==2 && m==1 && classweights[i][k] > 0 && weights[i][j][m] > 0) 
                                           conf = 10;
                                        if(k==5 && m==1 && classweights[i][k] > 0 && weights[i][j][m] > 0) 
                                           conf = 5;
                                        //if((k==10 && m==1 && classweights[i][k] > 0 && (classweights[i][10]-classweights[i][9]) > 0.1 && weights[i][j][m] > 0))
                                        //      (k==10 && m==2 && classweights[i][k] > 0 && weights[i][j][m] > 0))
                                           //conf = 2.5;
                                        //   conf = 5;
                                        if(k==11 && m==2 && classweights[i][k] > 0 && weights[i][j][m] > 0)
                                           conf = 10;
                                        if(k==10 && m==2 && classweights[i][10]>0 && (classweights[i][10] - classweights[i][9]) > 0 && weights[i][j][m] > 0)
                                           conf = 12.5;
                                        if(isrefs[i][j]==0 && k==11 && m==0 && (classweights[i][k]+weights[i][j][m])/2.0 != 1)
                                             conf = 0;
                                        //if(k==9 && m==2 && classweights[i][k]>0 && (classweights[i][9] - classweights[i][10]) > 0 && weights[i][j][m] > 0)
                                        //   conf = 5;
				        J1 += conf*jvars[auxind]*jointweights[jindex[i][0]][jindex[i][j]][k][m];
                                        acnt++;
                                        auxind++;
                                             
				}
			}
                        if(acnt > 0)
                           jexp1 += 0.45*J1*(1.0/(double)acnt);
                        //jexp1 += J1;
                         //jexp1 += J1;
                        /*acnt = 0;
                        GRBLinExpr J2=0;
			for(int k=0;k<4;k++)
			{
				for(int m=0;m<3;m++)
				{
					if(cjointweights[jindex[i][0]][jindex[i][j]][k][m]==0)
						continue;
					model.addConstr(2-gvars[i][0][k]-gvars[i][j][m]+jvars[auxind] >= 1,"JL3_"+itos(i)+"_"+itos(j)+"_"+itos(k)+"_"+itos(m));
					model.addConstr(gvars[i][0][k]+1-jvars[auxind] >= 1,"JL4_"+itos(i)+"_"+itos(j)+"_"+itos(k)+"_"+itos(m));
					model.addConstr(gvars[i][j][m]+1-jvars[auxind] >= 1,"JL5_"+itos(i)+"_"+itos(j)+"_"+itos(k)+"_"+itos(m));
				        //jexp1 += jvars[auxind]*cjointweights[jindex[i][0]][jindex[i][j]][k][m];
                                        if(k > 1 && m > 0)
				            J2 += jvars[auxind]*cjointweights[jindex[i][0]][jindex[i][j]][k][m];
                                        else
				            J2 += jvars[auxind]*cjointweights[jindex[i][0]][jindex[i][j]][k][m];
                                        acnt++;
                                        auxind++;
				}
			}
                        if(acnt > 0)
                            jexp1 += 0.0001*J2*(1.0/(double)acnt);*/
		}
                if(validvals > 0)
                    obj += 0.9*jexp1*(1.0/(double)validvals);
                 //obj += jexp1;
	}
    }
	for(int i=0;i<weights.size();i++)
	{
		GRBLinExpr cexpr = 0;
		for(int j=0;j<typevars;j++)
		{
			cexpr += classvars[i][j];
		}
		//cexpr += gvars[i][0][0] + gvars[i][0][3];
		model.addConstr(cexpr == 1.0,"C_"+itos(i));
		GRBLinExpr cexpr1 = 0;
		for(int j=1;j<9;j++)
			cexpr1 += classvars[i][j];
                cexpr1 += classvars[i][12]+classvars[i][13];
		model.addConstr(cexpr1==gvars[i][0][1],"CL_"+itos(i));
		//model.addConstr(cexpr1 - 1 - 100*gvars[i][0][1]<=0,"CL_"+itos(i));
		//model.addConstr(gvars[i][0][1] - 100*cexpr1<=0,"CLA_"+itos(i));
		GRBLinExpr cexpr2 = classvars[i][9] + classvars[i][11];
		model.addConstr(cexpr2==gvars[i][0][2],"CL1_"+itos(i));
		//model.addConstr(cexpr2 - 1 - 100*gvars[i][0][2]<=0,"CL1_"+itos(i));
		//model.addConstr(gvars[i][0][2] - 100*cexpr2<=0,"CL1A_"+itos(i));*/
		model.addConstr(gvars[i][0][0]==classvars[i][0],"CL2_"+itos(i));
		model.addConstr(gvars[i][0][3]==classvars[i][10],"CL3_"+itos(i));
		for(int j=0;j<weights[i].size();j++)
		{
		        int ntypes = numtrigtypes;
		        if(j > 0)
			     ntypes = numargtypes;
			GRBLinExpr expr = 0;
			for(int k=0;k<ntypes;k++)
			    expr += gvars[i][j][k];
			string s = "Sum_" + itos(i) + "_" + itos(j);
			model.addConstr(expr == 1.0, s);
		}
		//Not-None trig => at least 1 theme arg
		GRBLinExpr expr1 = gvars[i][0][0];
		for(int k=1;k<weights[i].size();k++)
		{
			expr1 += gvars[i][k][1];
		}
                string s = "c1_"+itos(i);
		model.addConstr(expr1 >= 1, s);
              
		for(int k=1;k<weights[i].size();k++)
                {
                    model.addConstr(2-gvars[i][0][1]-gvars[i][k][2] >= 1,"M_"+itos(i)+"_"+itos(k));
                }
		//None trig => none args
		/*for(int k=1;k<weights[i].size();k++)
		{
			GRBLinExpr expr1 = (1.0 - gvars[i][0][0]);
			expr1 += gvars[i][k][0];
			string sa = "c1A_"+itos(i)+"_"+itos(k);
			model.addConstr(expr1 >= 1, sa);
		}
		//none args => none trig
		GRBLinExpr expr2 = 0;
		for(int k=1;k<weights[i].size();k++)
		{
			expr2 += (1.0 - gvars[i][k][0]);
		}
		expr2 += gvars[i][0][0];
		model.addConstr(expr2 >= 1, "c1B_"+itos(i));
		//cause argument=>trigger is regulation
		for(int j=1;j<weights[i].size();j++)
		{
			model.addConstr(gvars[i][0][2] + gvars[i][0][3] >= gvars[i][j][2], "c8_"+itos(i)+"_"+itos(j));
		}*/
		/*if(trigrefs[i].size()!=0){	
		  //argthatistrigrefisnotnone => its trigger is regulation
		  for(int j=0;j<trigrefs[i].size()-1;j+=2)
		  {
			model.addConstr(gvars[trigrefs[i][j]][0][2] + gvars[trigrefs[i][j]][0][3]  + gvars[trigrefs[i][j]][trigrefs[i][j+1]][0] >= 1,"c7_"+itos(i)+"_"+itos(j));
		  }
                  GRBLinExpr e2 = 0;
		  for(int j=0;j<trigrefs[i].size()-1;j+=2)
		  {
                      e2 += gvars[trigrefs[i][j]][trigrefs[i][j+1]][0];
                      //model.addConstr(gvars[trigrefs[i][j]][trigrefs[i][j+1]][0]+1-gvars[i][0][0]>=1,"M_"+itos(i)+"_"+itos(j));
		  }
                  //model.addConstr(trigrefs[i].size()/2-e2-1000*(1-gvars[i][0][0]) <= 0,"CR_"+itos(i));
                  model.addConstr((e2+1-gvars[i][0][0]) >= 1,"CR_"+itos(i));
		}*/
	}
    /*GRBLinExpr tx = 0;
    for(int i=0;i<weights.size();i++)
    {
		//tx += 0.5*weights[i][0][0]*gvars[i][0][0]+1.3*weights[i][0][1]*gvars[i][0][1]+0.5*weights[i][0][2]*gvars[i][0][2]+0.3*weights[i][0][3]*gvars[i][0][3];
		tx += 0.145*weights[i][0][0]*gvars[i][0][0]+0.33*weights[i][0][1]*gvars[i][0][1]+0.145*weights[i][0][2]*gvars[i][0][2]+0.1*weights[i][0][3]*gvars[i][0][3];
    }
    obj += tx*(1.0/(double)weights.size()); */
    for(int i=0;i<weights.size();i++){
        GRBLinExpr ax = 0;
        for(int j=1;j<weights[i].size();j++){
             //ax += 0.8*weights[i][j][0]*gvars[i][j][0]+0.4*weights[i][j][1]*gvars[i][j][1]+1.2*weights[i][j][2]*gvars[i][j][2];
             //ax += 0.215*weights[i][j][0]*gvars[i][j][0]+0.123*weights[i][j][1]*gvars[i][j][1]+0.308*weights[i][j][2]*gvars[i][j][2];
             //BEST
             ax += 0.21*weights[i][j][0]*gvars[i][j][0]+0.12*weights[i][j][1]*gvars[i][j][1]+0.30*weights[i][j][2]*gvars[i][j][2];
             //ax += weights[i][j][0]*gvars[i][j][0]+weights[i][j][1]*gvars[i][j][1]+weights[i][j][2]*gvars[i][j][2];
        }
        obj += ax*(1.0/(double)(weights[i].size()-1)); 
    }
    GRBLinExpr typetx = 0;
    for(int i=0;i<weights.size();i++)
    {
                /*bool iscause = false;
                for(int j=1;j<weights[i].size();j++){
                   if(weights[i][j][2] > 0){
                     iscause=true;
                     break;
                  }
                }*/
		/*typetx += 0.5*classweights[i][0]*classvars[i][0]+classweights[i][1]*classvars[i][1]+classweights[i][2]*classvars[i][2]+
			3.75*classweights[i][3]*classvars[i][3]+classweights[i][4]*classvars[i][4]+2*classweights[i][5]*classvars[i][5]+
			classweights[i][6]*classvars[i][6]+classweights[i][7]*classvars[i][7]+classweights[i][8]*classvars[i][8]+
			classweights[i][9]*classvars[i][9]+classweights[i][10]*classvars[i][10]+classweights[i][11]*classvars[i][11];*/
                /*typetx += 0.145*classweights[i][0]*classvars[i][0]+0.262*classweights[i][1]*classvars[i][1]+0.262*classweights[i][2]*classvars[i][2]+
                        0.65*classweights[i][3]*classvars[i][3]+0.16*classweights[i][4]*classvars[i][4]+0.4*classweights[i][5]*classvars[i][5]+
                        0.262*classweights[i][6]*classvars[i][6]+0.262*classweights[i][7]*classvars[i][7]+0.262*classweights[i][8]*classvars[i][8]+
                        0.262*classweights[i][9]*classvars[i][9]+0.262*classweights[i][10]*classvars[i][10]+0.262*classweights[i][11]*classvars[i][11]+
						0.262*classweights[i][12]*classvars[i][12]+0.262*classweights[i][13]*classvars[i][13];*/
                //BEST
                /*if(iscause || classweights[i][11] > 0 || (classweights[i][1]-classweights[i][11]) < 0.1){
                typetx += 0.14*classweights[i][0]*classvars[i][0]+0.28*classweights[i][1]*classvars[i][1]+0.28*classweights[i][2]*classvars[i][2]+
                        0.85*classweights[i][3]*classvars[i][3]+0.16*classweights[i][4]*classvars[i][4]+0.4*classweights[i][5]*classvars[i][5]+
                        0.262*classweights[i][6]*classvars[i][6]+0.262*classweights[i][7]*classvars[i][7]+0.262*classweights[i][8]*classvars[i][8]+
                        0.262*classweights[i][9]*classvars[i][9]+0.262*classweights[i][10]*classvars[i][10]+0.28*classweights[i][11]*classvars[i][11]+
						0.262*classweights[i][12]*classvars[i][12]+0.262*classweights[i][13]*classvars[i][13];
                }
                else{*/
                typetx += 0.14*classweights[i][0]*classvars[i][0]+0.28*classweights[i][1]*classvars[i][1]+0.28*classweights[i][2]*classvars[i][2]+
                        0.85*classweights[i][3]*classvars[i][3]+0.16*classweights[i][4]*classvars[i][4]+0.4*classweights[i][5]*classvars[i][5]+
                        0.262*classweights[i][6]*classvars[i][6]+0.262*classweights[i][7]*classvars[i][7]+0.262*classweights[i][8]*classvars[i][8]+
                        0.262*classweights[i][9]*classvars[i][9]+0.262*classweights[i][10]*classvars[i][10]+0.262*classweights[i][11]*classvars[i][11]+
						0.262*classweights[i][12]*classvars[i][12]+0.262*classweights[i][13]*classvars[i][13];
                
                /*typetx += classweights[i][0]*classvars[i][0]+classweights[i][1]*classvars[i][1]+classweights[i][2]*classvars[i][2]+
                        classweights[i][3]*classvars[i][3]+classweights[i][4]*classvars[i][4]+classweights[i][5]*classvars[i][5]+
                        classweights[i][6]*classvars[i][6]+classweights[i][7]*classvars[i][7]+classweights[i][8]*classvars[i][8]+
                        classweights[i][9]*classvars[i][9]+classweights[i][10]*classvars[i][10]+classweights[i][11]*classvars[i][11]+
						classweights[i][12]*classvars[i][12]+classweights[i][13]*classvars[i][13];*/
    }
	obj += typetx*(1.0/(double)weights.size());

	model.setObjective(obj, GRB_MAXIMIZE);
	model.getEnv().set(GRB_IntParam_OutputFlag,0);
	model.optimize();

	vals.resize(weights.size());
	for(int i=0;i<weights.size();i++)
	{
		vals[i].resize(weights[i].size());
		double mx = classvars[i][0].get(GRB_DoubleAttr_X);
		int mxid = 0;
		for(int j=1;j<14;j++)
		{
			if(classvars[i][j].get(GRB_DoubleAttr_X) > mx)
			{
				mx = classvars[i][j].get(GRB_DoubleAttr_X);
				mxid = j;
			}
		}
		vals[i][0] = mxid;
		for(int j=1;j<weights[i].size();j++)
		{
			double amx;
			int amxid;
			for(int k=0;k<numargtypes;k++)
			{
				if(k==0)
				{
					amxid = 0;
					amx = gvars[i][j][k].get(GRB_DoubleAttr_X);
				}
				else if(gvars[i][j][k].get(GRB_DoubleAttr_X) > amx)
				{
					amxid = k;
					amx = gvars[i][j][k].get(GRB_DoubleAttr_X);
				}
			}
			vals[i][j] = amxid;
		}

	}
    /*for(int i=0;i<weights.size();i++)
    {
		vals[i].resize(weights[i].size());
		double mx = gvars[i][0][0].get(GRB_DoubleAttr_X);
		double mxid = 0;
		if(gvars[i][0][3].get(GRB_DoubleAttr_X) > mx)
		{
			mxid = 10;
			mx = gvars[i][0][3].get(GRB_DoubleAttr_X);
		}
		for(int j=0;j<typevars;j++)
		{
			if(classvars[i][j].get(GRB_DoubleAttr_X) > mx)
			{
				if(j<8)
				{
					mxid = j+1;
				}
				else
				{
					if(j==8)
					{
						mxid = 9;
					}
					else
					{
						mxid = 11;
					}
				}
				mx = classvars[i][j].get(GRB_DoubleAttr_X);
			}
		}
		vals[i][0] = mxid;
		for(int j=1;j<weights[i].size();j++)
		{
			double amx;
			int amxid;
			for(int k=0;k<numargtypes;k++)
			{
				if(k==0)
				{
					amxid = 0;
					amx = gvars[i][j][k].get(GRB_DoubleAttr_X);
				}
				else if(gvars[i][j][k].get(GRB_DoubleAttr_X) > amx)
				{
					amxid = k;
					amx = gvars[i][j][k].get(GRB_DoubleAttr_X);
				}
			}
			vals[i][j] = amxid;
		}
	}*/
    for(int i=0;i<weights.size();i++)
    {
        for(int j=0;j<weights[i].size();j++)
        {
	    delete[] gvars[i][j];
        }
	delete[] gvars[i];
    }
	delete[] gvars;
	for(int i=0;i<weights.size();i++)
		delete[] classvars[i];
	delete[] classvars;
	delete[] auxvars2;
	if(numjointvars > 0)
		delete[] jvars;
	if(includesoft1)
		delete[] refvars;
}



void jinfer(string filename)
{
        int errs = 0;
	ifstream flist(filename.c_str());
	while(flist.good())
	{
	string fname;
	getline(flist,fname);
	if(fname.size() < 1)
		continue;
        cout<<fname<<endl;
	ifstream infile(fname.c_str());
	string tmp = outdir+fname.substr(fname.find("/")+1);
	ofstream ofile(tmp.c_str());
	while(infile.good())
	{
		vector<vector<int> > trigrefs;
		vector<vector<vector<double> > > weights;
		vector<vector<int> > types;
		vector<vector<int> > jointindexes;
		vector<vector<double> > typewts;
		//parse file
		int numtrigs;
		infile >> numtrigs;
		if(numtrigs<=0 || numtrigs > 10000 )
		{
			string s;
			getline(infile,s);
			if(s.find("EOF")!=string::npos)
				break;
			else
				continue;
		}
		for(int i=0;i<numtrigs;i++)
		{
			vector<int> tmpref;
			vector<int> tmpaddr;
			vector<int> tmptp;
			vector<vector<double> > unitwts;
			vector<int> jindx;
			int num1;
			infile >> num1;
			if(num1==1)
			{
				//do not include
				int dum1;
				infile >> dum1;
				infile >> dum1;
				continue;
			}
			int tword;
			infile >> tword;
			vector<double> tunit(4);
			for(int j=0;j<tunit.size();j++)
				infile >> tunit[j];
			unitwts.push_back(tunit);
			int ttype;
			infile >> ttype;
			tmptp.push_back(ttype);
			jindx.push_back(tword);
			for(int j=1;j<num1;j++)
			{
				int adep;
				infile >> adep;
				vector<double> aunit(3);
				for(int k=0;k<aunit.size();k++)
					infile >> aunit[k];
				unitwts.push_back(aunit);
				int atype;
				infile >> atype;
				tmptp.push_back(atype);
				jindx.push_back(adep);
			}
			weights.push_back(unitwts);
			vector<int> tmprefs;
			trigrefs.push_back(tmprefs);
			types.push_back(tmptp);
			jointindexes.push_back(jindx);
		}
		//get the refs
		string s;
		getline(infile,s);
		getline(infile,s);
		if(s.size() > 1)
		{
			vector<string> parts;
			tokenize(s,parts,' ');
			for(int i=0;i<parts.size();i++)
			{
				if(parts[i].size()==0)
					continue;
				vector<int> refs;
				tointarr(parts[i],refs);
				trigrefs[refs[0]].push_back(refs[1]);
				trigrefs[refs[0]].push_back(refs[2]);
			}
		}
		for(int i=0;i<numtrigs;i++)
		{
			vector<double> tmp(14);
			for(int i=0;i<14;i++)
				infile >> tmp[i];
			typewts.push_back(tmp);
		}
		string s1;
		getline(infile,s1);
		vector<vector<int> > predids;
		ILPnew(weights,typewts,jointindexes,trigrefs,predids);
	        //ILPnoconstr(weights,typewts,jointindexes,trigrefs,predids);	
		for(int i=0;i<predids.size();i++)
		{
			for(int j=0;j<predids[i].size();j++)
			{
				if(j==0)
					ofile<<"T:"<<predids[i][j]<<endl;
				else
					ofile<<"A:"<<predids[i][j]<<endl;
				if(j==0)
					triggerconfusion[types[i][j]][predids[i][j]]++;
				else
					argconfusion[types[i][j]][predids[i][j]]++;
			}
		}
	}
	infile.close();
	ofile.close();
	}
	flist.close();
}
void loaddata()
{
	ifstream infile(mlnwtsfile);
	int totaltoks;
	infile >> totaltoks;
	for(int ii=0;ii<totaltoks;ii++)
	{
		int tw,aw;
		infile >> tw;
		infile >> aw;
		for(int i=0;i<12;i++)
		{
			for(int j=0;j<3;j++)
				infile >> jointweights[tw][aw][i][j];
		}
	}
	infile.close();
	if(includesoft1)
	{
		ifstream in1("refwts.dat");
		int ns;
		in1 >> ns;
		for(int i=0;i<ns;i++)
		{
			int tw,ad,n1;
			in1 >> tw;
			in1 >> ad;
			in1 >> n1;
			for(int j=0;j<n1;j++)
			{
				int a1,a2,cnt;
				double val;
				in1 >> a1;
				in1 >> a2;
				in1 >> val;
				outrefwts[tw][ad][a1][a2] = val;
				refindex[tw][ad].push_back(a1);
				refindex[tw][ad].push_back(a2);
			}
		}
		in1.close();
	}
	cout<<"Initialized weights..."<<endl;
}
void loadwts()
{
	/*
    ifstream infile("wts.dat");
    for(int i=0;i<totalvecsize;i++){
        infile >> avgweights[i];
    }
    infile.close();
    cout<<"lOADED WTS"<<endl;
	*/
}
};


int main(int argc, char* argv[])
{
	srand(time(NULL));
	if(argc < 4)
	{
		cout<<"Usage: jinfer softevidfilelisting outdir mlnwtsfile"<<endl;
		return -1;
	}
	//string s1(argv[2]);
	//stringstream st1(s1);
	//int includesoft1;
	//st1 >> includesoft1;
	LvrLearner lp;
	lp.includesoft1 = false;
	lp.outdir = argv[2];
	lp.mlnwtsfile = argv[3];
	//if(includesoft1==1)
	//	lp.includesoft1 = true;
 	lp.loaddata();
 	lp.jinfer(argv[1]);
	return 0;
}
