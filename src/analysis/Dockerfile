FROM rust:latest

workdir /usr/src/analysis
COPY . .

RUN apt-get update
RUN apt-get install -y gfortran
RUN cargo install --path .

CMD ["analysis"]
