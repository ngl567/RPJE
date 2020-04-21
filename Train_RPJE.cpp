// Last revision: 20191126
// rule confidence is: 0.7	change the confidence threshold: rule_path
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<omp.h>
using namespace std;


#define pi 3.1415926535897932384626433832795


map<vector<int>,string> path2s;  // path convert to string


map<pair<string,int>,double>  path_confidence;

bool L1_flag=1;

//normal distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
	// calculate the length of the vector
	double res=0;
/*	if (L1_flag)
		for (int i=0; i<a.size(); i++)
			res+=fabs(a[i]);
	else*/
	{
		for (int i=0; i<a.size(); i++)
			res+=a[i]*a[i];
		res = sqrt(res);
	}
	return res;
}

string version;
char buf[100000],buf1[100000],buf2[100000];
int relation_num,entity_num;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<pair<int, int>, pair<int, double>> rule2rel;		// used for path compositon by R2 rules
map<int, vector<pair<int, double> > > rel2rel;			// used for relations association by R1 rules
map<pair<int, int>, int> rule_ok;


vector<vector<pair<int,int> > > path;

class Train{

public:
	map<pair<int,int>, map<int,int> > ok;
    void add(int x,int y,int z, vector<pair<vector<int>,double> > path_list)
    {
	// add head entity: x, tail entity: y, relation: z, relation path: path_list, ok: 1 if the triple x-z-y added
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
	fb_path.push_back(path_list);
        ok[make_pair(x,z)][y]=1;
    }
    void pop()
    {
        fb_h.pop_back();
	fb_r.pop_back();
	fb_l.pop_back();
	fb_path.pop_back();
    }
    void run()
    {
        n = 100;
        rate = 0.001;
	regul = 0.01;
	cout<<"n="<<n<<' '<<"rate="<<rate<<endl;
	relation_vec.resize(relation_num);
		for (int i=0; i<relation_vec.size(); i++)
			relation_vec[i].resize(n);
        entity_vec.resize(entity_num);
		for (int i=0; i<entity_vec.size(); i++)
			entity_vec[i].resize(n);
        relation_tmp.resize(relation_num);
		for (int i=0; i<relation_tmp.size(); i++)
			relation_tmp[i].resize(n);
        entity_tmp.resize(entity_num);
		for (int i=0; i<entity_tmp.size(); i++)
			entity_tmp[i].resize(n);
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(entity_vec[i]);
        }

        bfgs();
    }

private:
    int n;
    double res;//loss function value
    double count,count1;//loss function gradient
    double rate;//learning rate
    double belta;
    double regul; //regulation factor
    int relrules_used;
    vector<int> fb_h,fb_l,fb_r;  // ID of the head entity, tail entity and relation
    vector<vector<pair<vector<int>,double> > >fb_path;   // all the relation paths
    vector<vector<double> > relation_vec,entity_vec;   // entity and relation embeddings to be learned
    vector<vector<double> > relation_tmp,entity_tmp;
    vector<vector<vector<double> > > R, R_tmp;
	
    double norm(vector<double> &a)
    {
        double x = vec_len(a);
        if (x>1)
        for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
            res+=x;
        return res;
    }

    void bfgs()
    {
	// training procedure
        double margin = 1,margin_rel = 1;
        cout<<"margin="<<' '<<margin<<"margin_rel="<<margin_rel<<endl;
        res=0;
        int nbatches=100;
        int nepoches = 500;
	cout<<"nbatches: "<<nbatches<<"\n";
	cout<<"nepoches: "<<nepoches<<"\n";
        int batchsize = fb_h.size()/nbatches;
	cout<<"The total number of triples is: "<<fb_h.size()<<"\n";
	cout<<"batchsize is: "<<batchsize<<"\n";
 	relation_tmp=relation_vec;
	entity_tmp = entity_vec;
	
        for (int epoch=0; epoch<nepoches; epoch++)
        {
        	res=0;
		int rules_used = 0;
		relrules_used = 0;
         	for (int batch = 0; batch<nbatches; batch++)
         	{
			int e1 = rand_max(entity_num);
         		for (int k=0; k<batchsize; k++)
         		{

					int entity_neg=rand_max(entity_num);
					int i=rand_max(fb_h.size());
					int e1 = fb_h[i], rel = fb_r[i], e2  = fb_l[i];
					
					int rand_tmp = rand()%100;
					if (rand_tmp<25)
					{
						while (ok[make_pair(e1,rel)].count(entity_neg)>0)
							entity_neg=rand_max(entity_num);
                        				train_kb(e1,e2,rel,e1,entity_neg,rel,margin);
					}
					else
					if (rand_tmp<50)
					{
						while (ok[make_pair(entity_neg,rel)].count(e2)>0)
							entity_neg=rand_max(entity_num);
	`			                        train_kb(e1,e2,rel,entity_neg,e2,rel,margin);
					}
					else
					{
						int rel_neg = rand_max(relation_num);
						while (ok[make_pair(e1,rel_neg)].count(e2)>0)
							rel_neg = rand_max(relation_num);
				                       	train_kb(e1,e2,rel,e1,e2,rel_neg,margin);
					}
					if (fb_path[i].size()>0)
					{
						// the training procedure of paths
						int rel_neg = rand_max(relation_num);
						while (ok[make_pair(e1,rel_neg)].count(e2)>0)
							rel_neg = rand_max(relation_num);
						for (int path_id = 0; path_id<fb_path[i].size(); path_id++)
						{
							vector<int> rel_path = fb_path[i][path_id].first;
							string  s = "";
							if (path2s.count(rel_path)==0)
							{
							    ostringstream oss;
								for (int ii=0; ii<rel_path.size(); ii++)
								{
									oss<<rel_path[ii]<<" ";
								}
							    s=oss.str();//
								path2s[rel_path] = s;
							}
							s = path2s[rel_path];

							double pr = fb_path[i][path_id].second;  // the reliability of the path
							double pr_path = 0;
							int rel_integ;
							double confi_integ = 0;
							double confi_path = 1;
							if (path_confidence.count(make_pair(s,rel))>0)
								pr_path = path_confidence[make_pair(s,rel)];
							pr_path = 0.99*pr_path + 0.01;
							if (rel_path.size() > 1){
							    for (int i = 0; i < rel_path.size(); i++){
							        if (rule2rel.count(make_pair(rel_path[i], rel_path[i+1])) > 0){
								    rules_used++;  // the amount of rules R2 used
                                                                    rel_integ = rule2rel[make_pair(rel_path[i], rel_path[i+1])].first;
                                                                    confi_integ = rule2rel[make_pair(rel_path[i], rel_path[i+1])].second;
								    confi_path = confi_path * confi_integ;
								    rel_path[i] = rel_integ;
								    for (int j = (i+1); j < (rel_path.size() - 1); j++){
								        rel_path[j] = rel_path[j+1];
								    }
								    rel_path.pop_back();
                                                        	}
							    }
							}

							train_path(rel,rel_neg,rel_path,2*margin,pr*pr_path);
						}
					}
			norm(relation_tmp[rel]);
            		norm(entity_tmp[e1]);
            		norm(entity_tmp[e2]);
            		norm(entity_tmp[entity_neg]);
			e1 = e2;
         		}
	            relation_vec = relation_tmp;
	            entity_vec = entity_tmp;
         	}
            cout<<"epoch:"<<epoch<<' '<<res<<endl;
	    cout<<"The number of R2 rules (rules of length 2) used in this epoch is: "<<rules_used<<"\n";
	    cout<<"The number of R1 rules (rules of length 1) used in this epoch is: "<<relrules_used<<"\n";
	    if (epoch>400 && (epoch+1)%100==0){
                int save_n = (epoch+1)/100;
                string serial = to_string(save_n);
                FILE* f2 = fopen(("./res/relation2vec_rule70_"+serial+".txt").c_str(),"w");
                FILE* f3 = fopen(("./res/entity2vec_rule70_"+serial+".txt").c_str(),"w");
                for (int i=0; i<relation_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
                    fprintf(f2,"\n");
                }
                for (int i=0; i<entity_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
                    fprintf(f3,"\n");
                }
                fclose(f2);
                fclose(f3);
                cout<<"Saving the training result succeed!"<<endl;
                }

	    }  // epoch
    }   // bfgs()
    double res1;
    double calc_kb(int e1,int e2,int rel)
   {
        double sum=0;
        for (int ii=0; ii<n; ii++)
		{
			double tmp = entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
	        if (L1_flag)
				sum+=fabs(tmp);
			else
				sum+=sqr(tmp);
		}
        return sum;
    }

    // calculate the similarity between two relations
    double calc_rule(int rel, int relpn){
	double sum = 0;
	for (int ii = 0; ii < n; ii++){
		double tmp = relation_vec[rel][ii] - relation_vec[relpn][ii];
		if (L1_flag)
			sum += fabs(tmp);
		else
			sum += sqr(tmp);
	}
        return sum;
    }

    void gradient_kb(int e1,int e2,int rel, double belta)
    {
        for (int ii=0; ii<n; ii++)
        {

            double x = 2*(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel][ii]-=belta*rate*x;
            entity_tmp[e1][ii]-=belta*rate*x;
            entity_tmp[e2][ii]+=belta*rate*x;
        }
    }

    // gradient of relation association
    void gradient_rule(int rel1, int rel2, double belta)
    {
	for (int ii=0; ii<n; ii++){
		double x = 2*(relation_vec[rel1][ii] - relation_vec[rel2][ii]);
		if (L1_flag)
			if (x>0)
				x = 1;
			else
				x = -1;
		relation_tmp[rel1][ii] += belta*rate*x;
		relation_tmp[rel2][ii] -= belta*rate*x;
	}
    }

    double calc_path(int r1,vector<int> rel_path)
    {
    // calculate the similarity between path and relation pair
        double sum=0;
        for (int ii=0; ii<n; ii++)
		{
			double tmp = relation_vec[r1][ii];

			for (int j=0; j<rel_path.size(); j++)
				tmp-=relation_vec[rel_path[j]][ii];
	        if (L1_flag)
				sum+=fabs(tmp);
			else
				sum+=sqr(tmp);
		}
        return sum;
    }
    void gradient_path(int r1,vector<int> rel_path, double belta)
    {
        for (int ii=0; ii<n; ii++)
        {

			double x = relation_vec[r1][ii];
			for (int j=0; j<rel_path.size(); j++)
				x-=relation_vec[rel_path[j]][ii];
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[r1][ii]+=belta*rate*x;
			for (int j=0; j<rel_path.size(); j++)
            	relation_tmp[rel_path[j]][ii]-=belta*rate*x;

        }
    }
    void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,double margin)
    {
        double sum1 = calc_kb(e1_a,e2_a,rel_a);
        double sum2 = calc_kb(e1_b,e2_b,rel_b);
	double lambda_rule = 3;
	double marginrule = 1;
        if (sum1+margin>sum2)
        {
        	res+=margin+sum1-sum2;
        	gradient_kb(e1_a, e2_a, rel_a, -1);
		gradient_kb(e1_b, e2_b, rel_b, 1);
        }
	if (rel2rel.count(rel_a) > 0)
	{
	    for (int i = 0; i < rel2rel[rel_a].size(); i++){
		int rel_rpos = rel2rel[rel_a][i].first;
		double rel_pconfi = rel2rel[rel_a][i].second;
		double sum_pos = calc_rule(rel_a, rel_rpos);
		int rel_rneg = rand_max(relation_num);
		while (rule_ok.count(make_pair(rel_a, rel_rneg)) > 0)
			rel_rneg = rand_max(relation_num);
		double sum_neg = calc_rule(rel_a, rel_rneg);
		if (rel_pconfi*sum_pos + marginrule > sum_neg){
			res += margin + rel_pconfi*sum_pos - sum_neg;
			gradient_rule(rel_a, rel_rpos, -lambda_rule);
			gradient_rule(rel_a, rel_rneg, lambda_rule);
		}
		norm(relation_tmp[rel_a]);
		norm(relation_tmp[rel_rpos]);
		norm(relation_tmp[rel_rneg]);
		relrules_used++;
	    }
	}
    }
    void train_path(int rel, int rel_neg, vector<int> rel_path, double margin,double x)
    {
        double sum1 = calc_path(rel,rel_path);
        double sum2 = calc_path(rel_neg,rel_path);
	double lambda = 1;
        if (sum1+margin>sum2)
        {
        	res+=x*lambda*(margin+sum1-sum2);
        	gradient_path(rel,rel_path, -x*lambda);
		gradient_path(rel_neg,rel_path, x*lambda);
        }
    }

};

Train train;
void prepare()
{
    FILE* f1 = fopen("./data/entity2id.txt","r");
	FILE* f2 = fopen("./data/relation2id.txt","r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		id2relation[x+1345] = "-"+st;
		relation_num++;
	}
	FILE* f_kb = fopen("./data/train_pra.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
	{
        	string s1=buf;
        	fscanf(f_kb,"%s",buf);
	        string s2=buf;
	        if (entity2id.count(s1)==0)
        	{
            		cout<<"miss entity:"<<s1<<endl;
        	}
	        if (entity2id.count(s2)==0)
        	{
	            	cout<<"miss entity:"<<s2<<endl;
        	}
	        int e1 = entity2id[s1];
        	int e2 = entity2id[s2];
	        int rel;
		fscanf(f_kb,"%d",&rel);
		fscanf(f_kb,"%d",&x);
		vector<pair<vector<int>,double> > b;
		b.clear();
		for (int i = 0; i<x; i++)
		{
			int y,z;
			vector<int> rel_path;
			rel_path.clear();
			fscanf(f_kb,"%d",&y);
			for (int j=0; j<y; j++)
			{
				fscanf(f_kb,"%d",&z);
				rel_path.push_back(z);
			}
			double pr;
			fscanf(f_kb,"%lf",&pr);
			b.push_back(make_pair(rel_path,pr));
		}
        	train.add(e1,e2,rel,b);
	}
	train.pop();
	relation_num*=2;
   
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
	
	FILE* f_confidence = fopen("./data/confidence.txt","r");
	while (fscanf(f_confidence,"%d",&x)==1)
	{
		string s = "";
		for (int i=0; i<x; i++)
		{
			fscanf(f_confidence,"%s",buf);
			s = s + string(buf)+" ";
		}
		fscanf(f_confidence,"%d",&x);
		for (int i=0; i<x; i++)
		{
			int y;
			double pr;
			fscanf(f_confidence,"%d%lf",&y,&pr);
		//	cout<<s<<' '<<y<<' '<<pr<<endl;
			path_confidence[make_pair(s,y)] = pr;
		}
	}
	fclose(f_confidence);
    fclose(f_kb);

    cout<<"Load all the R1 rules.\n";
    int count_rules = 0;
    FILE* f_rule1 = fopen("./data/rule/rule_relation70.txt","r");
	int rel1, rel2, rel3;
	double confi;
        while (fscanf(f_rule1,"%d", &rel1)==1)
        {
                fscanf(f_rule1, "%d%lf", &rel2, &confi);
		rel2rel[rel1].push_back(make_pair(rel2, confi));
		rule_ok[make_pair(rel1, rel2)] = 1;
		count_rules++;
        }
    fclose(f_rule1);

    cout<<"Loading all the R2 rules.\n";
    FILE* f_rule2 = fopen("./data/rule/rule_path70.txt","r");
        while (fscanf(f_rule2,"%d%d", &rel1 ,&rel2)==2)
        {
                fscanf(f_rule2, "%d%lf", &rel3, &confi);
                rule2rel[make_pair(rel1, rel2)] = make_pair(rel3, confi);
		count_rules++;
        }
	cout<<"The confidence of rules is: 0.7"<<"\n";
        cout<<"The total number of rules is: "<<count_rules<<"\n";

    fclose(f_rule2);
}

int main(int argc,char**argv)
{

	cout << "Start to prepare!\n";
        prepare();
	cout << "Prepare Success!\n";
        cout << "Start Training!\n";
        train.run();
	cout << "Training finished.\n"
}
 
