#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
int main(int argc, char **argv) {
    for (int i=1; i<argc; ++i) {
        FILE *f=fopen(argv[i],"rb"); if (!f) return 1;
        uint32_t words[16384]; size_t n; uint64_t h=UINT64_C(0xcbf29ce484222325),bytes=0;
        while ((n=fread(words,sizeof(uint32_t),16384,f))) {
            for (size_t j=0;j<n;++j) { h^=words[j]; h*=UINT64_C(0x100000001b3); }
            bytes+=4*n;
        }
        if (ferror(f)) return 2;
        fclose(f); printf("%016" PRIx64 " %" PRIu64 " %s\n",h,bytes,argv[i]);
    }
}
