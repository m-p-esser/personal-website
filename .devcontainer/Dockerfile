################################
# Install Linux OS
################################

FROM debian:bookworm-slim

################################
# Install Linux packages
################################

# Set the environment variable for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

RUN \
   # Update
   apt-get update -y && \
   # Install dependencies
   apt-get install unzip wget curl make sudo hugo -y