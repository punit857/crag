$ErrorActionPreference = "Stop"

Write-Host "1. Creating secrets from local.env..."
# This reads your local env and creates a secure OpenShift Secret without printing the keys.
oc create secret generic crag-secrets --from-env-file=.env --dry-run=client -o yaml | oc apply -f -

Write-Host "`n2. Applying OpenShift YAML configurations..."
$manifests = @"
# REDIS
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--save", ""]
        resources:
          requests:
            cpu: 50m
            memory: 64Mi
          limits:
            cpu: 200m
            memory: 128Mi
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  ports:
  - port: 6379
  selector:
    app: redis
---
# QDRANT
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qdrant-storage
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
spec:
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.10.1
        ports:
        - containerPort: 6333
        - containerPort: 6334
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 1Gi
        volumeMounts:
        - name: storage
          mountPath: /qdrant/storage
        - name: snapshots
          mountPath: /qdrant/snapshots
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: qdrant-storage
      - name: snapshots
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
spec:
  ports:
  - name: http
    port: 6333
  - name: grpc
    port: 6334
  selector:
    app: qdrant
---
# API BUILD & IMAGE STREAM
apiVersion: image.openshift.io/v1
kind: ImageStream
metadata:
  name: crag-api
---
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: crag-api-build
spec:
  source:
    git:
      uri: https://github.com/punit857/crag.git
      ref: main
    type: Git
  strategy:
    dockerStrategy:
      dockerfilePath: Dockerfile
  output:
    to:
      kind: ImageStreamTag
      name: crag-api:latest
---
# API DEPLOYMENT
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crag-api
  annotations:
    # This automatically triggers a redeployment whenever a new image is built
    image.openshift.io/triggers: '[{"from":{"kind":"ImageStreamTag","name":"crag-api:latest"},"fieldPath":"spec.template.spec.containers[?(@.name==\"api\")].image"}]'
spec:
  selector:
    matchLabels:
      app: crag-api
  template:
    metadata:
      labels:
        app: crag-api
    spec:
      enableServiceLinks: false
      containers:
      - name: api
        image: crag-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: crag-secrets
        env:
        - name: QDRANT_USE_LOCAL
          value: "false"
        - name: QDRANT_HOST
          value: "qdrant"
        - name: QDRANT_PORT
          value: "6333"
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
        - name: RATE_LIMIT_DAILY_GLOBAL
          value: "10"
        resources:
          requests:
            cpu: 200m
            memory: 1Gi
          limits:
            cpu: "1"
            memory: 2.5Gi
        volumeMounts:
        - name: models
          mountPath: /models
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 12
      volumes:
      - name: models
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: crag-api
spec:
  ports:
  - port: 8000
  selector:
    app: crag-api
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: crag-api-route
spec:
  to:
    kind: Service
    name: crag-api
  port:
    targetPort: 8000
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
"@

$manifests | oc apply -f -

Write-Host "`n3. Starting OpenShift build from GitHub (this pulls your latest Dockerfile)..."
oc start-build crag-api-build --follow

Write-Host "`n4. Waiting for API rollout (FastEmbed downloading models, can take ~3-4 mins)..."
oc rollout status deployment/crag-api --watch=true

Write-Host "`n5. Fetching your public URL..."
$url = oc get route crag-api-route -o jsonpath='{.spec.host}'
Write-Host "========================================"
Write-Host "DEPLOYMENT COMPLETE!"
Write-Host "Your API is live at: https://$url"
Write-Host "========================================"