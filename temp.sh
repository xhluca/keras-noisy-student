# export NAME="noisy_student_efficientnet-l2"
# export MODEL="efficientnet-l2"
# export URL="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-l2.tar.gz"

export NAME="noisy_student_efficientnet-b0"
export MODEL="efficientnet-b0"
export URL="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b0.tar.gz"


wget $URL -P tf_weights
mkdir -p converted_weights
mkdir -p tf_weights

tar xvf "tf_weights/${NAME}.tar.gz" -C tf_weights

python load_efficientnet.py \
    --source ./efficientnet_tf \
    --model_name  $MODEL \
    --tf_checkpoint "./tf_weights/${NAME}" \
    --output_file "./converted_weights/${NAME}"