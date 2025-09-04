# Stage 1 — build
FROM node:18-alpine AS build

# Set working directory
WORKDIR /manualmind_client

# Install build deps
# Copy package files first to leverage Docker cache
COPY package*.json ./
# Use npm ci for reproducible builds (falls back to npm install if package-lock missing)
RUN npm ci --prefer-offline --no-audit --progress=false

# Copy rest of the sources
COPY . .

# Provide a build-time arg to set the API base URL (defaults to http://localhost:8000)
ARG REACT_APP_API_BASE_URL=http://localhost:8000
ENV REACT_APP_API_BASE_URL=${REACT_APP_API_BASE_URL}

# Build app (expects "build" script in package.json)
RUN npm run build

# Stage 2 — production image with nginx
FROM nginx:stable-alpine

# Remove default nginx website
RUN rm -rf /usr/share/nginx/html/*

# Copy built static files from previous stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy a custom nginx config to enable SPA fallback
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Run nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
