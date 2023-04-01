FROM rust:latest

RUN apt-get update
RUN apt-get install -y gfortran

RUN USER=root cargo new --bin analysis
workdir /analysis

COPY ./Cargo.toml ./Cargo.toml

RUN cargo build --release
RUN rm src/*.rs

COPY ./src ./src 

RUN rm ./target/release/deps/analysis*
RUN cargo install --path .

CMD ["analysis"]
