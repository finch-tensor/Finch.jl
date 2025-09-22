#!/bin/bash
#set -xe
#SBATCH -N 1
#SBATCH -t 360

dir=./sam-outputs
dot_dir=$dir/dot
png_dir=$dir/png
taco_exe=./taco/build/bin/taco

GEN_KERNEL_NAMES=(
  matmul_kij
  matmul_kji
  matmul_ikj
  matmul_jki
  matmul_ijk
  matmul_jik
  mat_elemmul
  mat_elemadd
  mat_identity
  mat_identity_dense
  mat_vecmul_ij
  mat_vecmul_ji
  vec_elemmul
  vec_identity
  vec_elemadd
  vec_scalar_mul
  tensor3_elemmul
  tensor3_identity
  tensor3_identity_dense
  tensor3_elemadd
  tensor3_innerprod
  tensor3_ttv
  tensor3_ttm
  mat_sddmm
  mat_mattransmul
  mat_residual
  mat_elemadd3
  tensor3_mttkrp
  vec_spacc_simple
  mat_spacc_simple
  vec_sd_compression_WRONG
  vec_ds_compression_WRONG
  vec_sub
  mat_mask_tri
  mat_mask_tri_partial
  mat_vecmul_iter
  mat_vecmul_iter_short
  tensor3_website_expr
  mat_mask_tri_DA3sum_final
  mat_mask_tri_DA3_final
)

HAND_KERNEL_NAMES=(
  mat_residual_HAND
  mat_mattransmul_HAND
  vec_sd_compression_HAND
  vec_ds_compression_HAND
)

TACO_ARGS=(
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss:1,0 -f=C:ss -s=reorder(k,i,j)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss:1,0 -f=B:ss:1,0 -f=C:ss -s=reorder(k,j,i)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss -f=C:ss -s=reorder(i,k,j)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss:1,0 -f=B:ss:1,0 -f=C:ss:1,0 -s=reorder(j,k,i)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss -f=C:ss:1,0  -s=reorder(i,j,k)"
  "X(i,j)=B(i,k)*C(k,j) -f=X:ss:1,0 -f=B:ss -f=C:ss:1,0  -s=reorder(j,i,k)"
  "X(i,j)=B(i,j)*C(i,j) -f=X:ss -f=B:ss -f=C:ss"
  "X(i,j)=B(i,j)+C(i,j) -f=X:ss -f=B:ss -f=C:ss"
  "X(i,j)=B(i,j) -f=X:ss -f=B:ss"
  "X(i,j)=B(i,j) -f=X:dd -f=B:dd"
  "x(i)=B(i,j)*c(j) -f=x:s -f=B:ss -f=c:s"
  "x(i)=B(i,j)*c(j) -f=x:s -f=B:ss:1,0 -f=c:s -s=reorder(j,i)"
  "x(i)=b(i)*c(i) -f=x:s -f=b:s -f=c:s"
  "x(i)=b(i) -f=x:s -f=b:s"
  "x(i)=b(i)+c(i) -f=x:s -f=b:s -f=c:s"
  "x(i)=b*c(i) -f=x:s -f=c:s"
  "X(i,j,k)=B(i,j,k)*C(i,j,k) -f=X:sss -f=B:sss -f=C:sss"
  "X(i,j,k)=B(i,j,k) -f=X:sss -f=B:sss"
  "X(i,j,k)=B(i,j,k) -f=X:ddd -f=B:ddd"
  "X(i,j,k)=B(i,j,k)+C(i,j,k) -f=X:sss -f=B:sss -f=C:sss"
  "x=B(i,j,k)*C(i,j,k) -f=B:sss -f=C:sss"
  "X(i,j)=B(i,j,k)*c(k) -f=X:ss -f=B:sss -f=c:s"
  "X(i,j,k)=B(i,j,l)*C(k,l) -f=X:sss -f=B:sss -f=C:ss"
  "X(i,j)=B(i,j)*C(i,k)*D(k,j) -f=X:ss -f=B:ss -f=C:dd -f=D:dd:1,0 -s=reorder(i,j,k)"
  "x(i)=b*C(j,i)*d(j)+e*f(i) -f=x:s -f=C:ss:1,0 -f=d:s -f=f:s"
  "x(i)=b(i)-C(i,j)*d(j) -f=x:s -f=C:ss -f=b:s -f=d:s"
  "X(i,j)=B(i,j)+C(i,j)+D(i,j) -f=X:ss -f=B:ss -f=C:ss -f=D:ss"
  "X(i,j)=B(i,k,l)*C(j,k)*D(j,l) -f=X:ss -f=B:sss -f=C:ss -f=D:ss" 
  "x(j)=B(i,j) -f=x:s -f=B:ss"
  "X(j,k)=B(i,j,k) -f=X:ss -f=B:sss"
  "x(i)=b(i) -f=b:s -f=x:d"
  "x(i)=b(i) -f=b:d -f=x:s"
  "x(i)=b(i)-c(i) -f=b:s -f=c:s"
  "x=B(i,j)*C(i,k)*D(k,j) -f=B:ss -f=C:ss -f=D:ss:1,0 -s=reorder(i,j,k)"
  "X(i,j)=B(i,j)*C(i,k)*D(k,j) -f=X:ss -f=B:ss -f=C:ss -f=D:ss:1,0 -s=reorder(i,j,k)"
  "x(i)=B(i,j)*C(j,k)*D(k,l)*E(l,m)*f(m) -f=x:s -f=B:ss -f=C:ss -f=D:ss -f=E:ss -f=f:s -s=reorder(i,j,k,l,m)"
  "x(i)=B(i,j)*C(j,k)*d(k) -f=x:s -f=B:ss -f=C:ss -f=d:s -s=reorder(i,j,k)"
  "x=B(i)*C(j)*D(i,j,k)*E(j,l)*F(l,m,n) -f=B:s -f=C:s -f=D:sss -f=E:ss -f=F:sss -s=reorder(i,j,k,l,m,n)"
  "x=Diag(i,l)*(B1(i,j)*B2(j,k)*B3(k,l)) -f=Diag:ss -f=B1:ss -f=B2:ss -f=B3:ss:1,0 -s=reorder(i,l,j,k)"
  "X(i,l)=Diag(i,l)*(B1(i,j)*B2(j,k)*B3(k,l)) -f=X:ss -f=Diag:ss -f=B1:ss -f=B2:ss -f=B3:ss:1,0 -s=reorder(i,l,j,k)"
)

mkdir -p $dir
mkdir -p $dot_dir
mkdir -p $png_dir

for i in ${!GEN_KERNEL_NAMES[@]}; do
    name=${GEN_KERNEL_NAMES[$i]}
    args=${TACO_ARGS[$i]}

    $taco_exe $args --print-sam-graph="$dot_dir/$name.gv"
    dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png
    echo "Generating sam for $name to $dir"
done

for i in ${!HAND_KERNEL_NAMES[@]}; do
    name=${HAND_KERNEL_NAMES[$i]}

    dot -Tpng $dot_dir/$name.gv -o $png_dir/$name.png
    echo "Generating sam for $name to $dir"
done

