#include <cstdio>
#include <queue>
#include <cstring>
#include<algorithm>
#include <string>
#define MAX 3

using namespace std ;

struct Node{
	int map[MAX][MAX] ,hash;
	int f,g,h;
	int x,y;
	/*bool operator<(const Node &n)const
	{
		return f>n.f ;
	}*/
	bool operator<(const Node n1) const{     //优先队列第一关键字为h,第二关键字为g
		return h!=n1.h?h>n1.h:g>n1.g;
	}

	bool check()
	{
		if(x<0||y<0 || x>=MAX||y>=MAX)
		{
			return false ;
		}
		return true ;
	}
};
const int HASH[9]={1,1,2,6,24,120,720,5040,40320};   //HASH的权值  
const int dir[4][2]={1,0,-1,0,0,-1,0,1} ;
int visited[400000] ;
int pre[400000] ;
int des = 322560 ;
int getHash(Node n)
{
	int oth[MAX*4] , k = 0;
	for(int i = 0 ; i < MAX ; ++i)
	{ 
		for(int j = 0 ; j < MAX ; ++j)
		{
			oth[k++] = n.map[i][j] ;
		}
	}
	int result = 0 ;
	for(int i = 0 ; i < 9 ; ++i)
	{
		int count = 0 ; 
		for(int j = 0 ; j < i ; ++j)
		{
			if(oth[i]<oth[j])
			{
				count++;
			}
		}
		result += count*HASH[i] ;
	}
	return result ;
}

int getH(Node n)
{
	int result = 0 ;
	for(int i = 0 ; i < MAX ; ++i)
	{
		for(int j = 0 ; j < MAX ; ++j)
		{
			if(n.map[i][j])
			{
				int x = (n.map[i][j]-1)/3 , y = (n.map[i][j]-1)%3 ;
				result += abs(x-i)+abs(y-j) ;
			}
		}
	}
	return result ;
}

bool judge(Node n)
{
	int oth[MAX*4] , k = 0;
	for(int i = 0 ; i < MAX ; ++i)
	{ 
		for(int j = 0 ; j < MAX ; ++j)
		{
			oth[k++] = n.map[i][j] ;
		}
	}
	int result = 0 ;
	for(int i = 0 ; i < 9 ; ++i)
	{
		for(int j = i+1 ; j < 9 ; ++j)
		{
			if(oth[i]&&oth[j]&&oth[i]>oth[j])
			{
				++result;
			}
		}
	}
	return !(result&1) ;
}

void AStar(Node start)
{
	priority_queue<Node> p;
	p.push(start);
	while(!p.empty())
	{
		Node n = p.top();
		p.pop();
		for(int i = 0 ; i < 4 ; ++i)
		{
			Node next = n;
			next.x += dir[i][0];
			next.y += dir[i][1];
			if(!next.check())
			{
				continue ;
			}
			swap(next.map[next.x][next.y],next.map[n.x][n.y]) ;
			next.hash = getHash(next) ;
			if(visited[next.hash] == -1)
			{
				next.h = getH(next);  //estimation of the spent estimation
				next.g++;
				next.f = next.g+next.h;
				pre[next.hash] = n.hash;
				p.push(next);
				visited[next.hash] = i;	//i代表方向 
			}
			if(next.hash == des)
			{
				return ;
			}
		}
	}
}

void print()
{
	int next = des ;
	string ans;
	ans.clear() ;
	while(pre[next]!=-1)
	{
		switch(visited[next])
		{
			case 0 : ans += 'd' ; break ;
			case 1 : ans += 'u' ; break ;
			case 2 : ans += 'l' ; break ;
			case 3 : ans += 'r' ; break ;
			default : break ; 
		}
		next = pre[next] ;
	}
	int len = ans.size() ;
	for(int i = len-1 ; i >=0 ; --i)
	{
		putchar(ans[i]) ;
	}
	puts("");
}

int main()
{
	char str[100] ;
	while(gets(str) != NULL)
	{
		Node t ;
		memset(visited, -1, sizeof(visited)) ;
		memset(pre, -1, sizeof(pre)) ;
		int k = 0 ,i = 0;
		while(str[k] != '\0')
		{
			if(str[k]>'0'&&str[k]<='9')
			{
				t.map[i/3][i%3] = str[k]-'0' ;
				++i ;
			}
			else if(str[k] == 'x')
			{
				t.x = i/3 ;
				t.y = i%3 ;
				t.map[i/3][i%3] = 0 ;
				++i ;
			}
			++k ;
		}
		t.hash=getHash(t);
		visited[t.hash] = -2 ;
		t.g = 0 ;
		t.h = getH(t);
		t.f = t.g+t.h;
		if(!judge(t))
		{
			printf("unsolvable\n"); 
			continue ;
		}
		if(t.hash == des)
		{
			puts(""); 
			continue ;
		}
		AStar(t) ;
		print();
	}
	return 0;
}


'''
// A* Search Algorithm
1.  Initialize the open list
2.  Initialize the closed list
    put the starting node on the open 
    list (you can leave its f at zero)

3.  while the open list is not empty
    a) find the node with the least f on 
       the open list, call it "q"

    b) pop q off the open list
  
    c) generate q's 8 successors and set their 
       parents to q
   
    d) for each successor
        i) if successor is the goal, stop search
          successor.g = q.g + distance between 
                              successor and q
          successor.h = distance from goal to 
          successor (This can be done using many 
          ways, we will discuss three heuristics- 
          Manhattan, Diagonal and Euclidean 
          Heuristics)
          
          successor.f = successor.g + successor.h

        ii) if a node with the same position as 
            successor is in the OPEN list which has a 
           lower f than successor, skip this successor

        iii) if a node with the same position as 
            successor  is in the CLOSED list which has
            a lower f than successor, skip this successor
            otherwise, add  the node to the open list
     end (for loop)
  
    e) push q on the closed list
    end (while loop)
'''
