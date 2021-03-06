# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:/storage/home/monishaj/.local/lib/python2.7/site-packages:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/storage/home/monishaj/.local/lib/python2.7/site-packages:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source activate py27

# Change to the directory in which your code is present
cd /storage/home/monishaj/image_task
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is the MNIST Tensorflow example.
python -u train.py &> out