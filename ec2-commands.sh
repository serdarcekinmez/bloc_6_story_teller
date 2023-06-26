sudo yum update -y
sudo yum install docker -y
echo -e "sudo service docker start" >> .bashrc
sudo usermod -a -G docker ec2-user
sudo yum install git -y
# logout and login again
touch docker-compose.yaml
vi docker-compose.yaml
# copy paste your local docker-compose.yaml file
sudo curl -SL https://github.com/docker/compose/releases/download/v2.18.1/docker-compose-linux-aarch64 -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
docker-compose --version
docker-compose up --detach