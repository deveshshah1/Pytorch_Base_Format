# Sample Dockerfile for containerizing this code
# First adjust code. All input/ouput files should be set to "./Pytorch_Base_Format/out.txt" instead of "out.txt". This will allow for volume mounting from docker
# Then organize code in following format
# /Users/.../CodeFolder/
# 	Dockerfile
# 	Pytorch_Base_Format


# To Build the Image Using Sample Paths:
# docker image build -t pytorch:base_format_img /Users/.../CodeFolder

# To Run Image first enter the Pytorch_Base_Format directory. Then run:
# docker run -v $PWD:/Pytorch_Base_Format/ pytorch:base_format_img

FROM pytorch/pytorch:latest
WORKDIR /
COPY Pytorch_Base_Format /Pytorch_Base_Format
RUN pip install xlsxwriter tensorboard tqdm scikit-learn seaborn numpy PIL pandas matplotlib
CMD ["python", "./Pytorch_Base_Format/main.py"]
