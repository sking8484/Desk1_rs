# Create chef image that installs cargo chef
FROM rust:1.72.0-bookworm as planner
WORKDIR /app
RUN cargo install cargo-chef
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM rust:1.72.0-bookworm AS cacher
WORKDIR /app
RUN cargo install cargo-chef
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching docker layer!
RUN cargo chef cook --release --recipe-path recipe.json

# Copy everything in current app
FROM rust:1.72.0-bookworm as builder 
COPY . /app

# Set the working directory to the app folder that we just copied everything in to.
WORKDIR /app

COPY --from=cacher /app/target target
COPY --from=cacher /usr/local/cargo /usr/local/cargo

# Build the image using release tag
RUN cargo build --release

# Pull down small debian image hosted on google container repo
FROM rust:1.72.0-bookworm AS runtime
ARG function

# Copy release build from builder image
COPY --from=builder /app/target/release/$function /app/programRunner
WORKDIR /app

CMD ["./programRunner"]
