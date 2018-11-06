#include <stdio.h>
int main(void)
{
    int a, b;
    long long result;
    while (scanf("%d %d", &a, &b)==2) {
        result = a + b;
        printf("%d\n",a+b);
    }
    return 0;
}
