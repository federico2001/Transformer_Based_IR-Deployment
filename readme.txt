RUN IN PRODUCTION 

docker build --no-cache -t ir_deployment .

docker run -p 5000:5000 ir_deployment


