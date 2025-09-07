# 🚂 Railway Setup Guide - Farmacia IA

## 📋 Pasos Detallados para Deploy

### 1. Crear Proyecto en Railway

1. **Ir a Railway**: https://railway.app
2. **Login** con GitHub/Google/Email
3. **New Project** → **Empty Project**
4. Nombrar proyecto: `farmacia-ia`

### 2. Agregar Servicios

#### A. PostgreSQL Database
1. **+ New** → **Database** → **Add PostgreSQL**
2. Nombre: `farmacia-postgres`
3. ✅ Se crea automáticamente `DATABASE_URL`

#### B. Redis Cache
1. **+ New** → **Database** → **Add Redis**  
2. Nombre: `farmacia-redis`
3. ✅ Se crea automáticamente `REDIS_URL`

#### C. Django Application
1. **+ New** → **GitHub Repo** (si tienes repo)
2. O **+ New** → **Empty Service** → **Deploy from local**
3. Nombre: `farmacia-django`

### 3. Configurar Django Service

#### Build Settings
- **Settings** → **Source**
- **Build Command**: (vacío - usa Docker)
- **Dockerfile Path**: `docker/Dockerfile.django`
- **Docker Context**: `.`

#### Deploy Settings  
- **Settings** → **Deploy**
- **Health Check Path**: `/health/`
- **Health Check Timeout**: `300`
- **Restart Policy**: `ON_FAILURE`
- **Custom Start Command**: (vacío - usa CMD del Dockerfile)

### 4. Variables de Entorno

#### Variables Críticas (OBLIGATORIAS)
```bash
SECRET_KEY=tu-clave-secreta-muy-larga-y-aleatoria-aqui
DEBUG=False
ALLOWED_HOSTS=*.railway.app,*.up.railway.app
```

#### Variables de IA
```bash
OPENAI_API_KEY=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
QDRANT_URL=https://07fa4f33-60b8-4833-ba9c-4334d3383856.us-west-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzgxMTk2NDIyfQ.Y1Nzow0mQJ9kf459Y_xzdRWvcTtgfMAKBO272xrEejk
QDRANT_COLLECTION_NAME=MiColeccionRag
```

#### Variables de LangSmith
```bash
LANGCHAIN_API_KEY=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
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=pr-frosty-floozie-91
LANGCHAIN_TRACING_V2=true
```

#### Variables de Seguridad
```bash
ENCRYPTION_KEY=dGVzdF9lbmNyeXB0aW9uX2tleV8zMl9ieXRlcw==
LOG_LEVEL=INFO
```

### 5. Deploy Process

1. **Subir código** (si usas Empty Service):
   - Comprimir proyecto en ZIP
   - Arrastrar a Railway
   - O conectar GitHub repo

2. **Trigger Deploy**:
   - Railway detecta cambios automáticamente
   - O hacer clic en **Deploy**

3. **Monitorear**:
   - Ver **Deployments** tab
   - Revisar **Logs** en tiempo real
   - Verificar **Metrics**

### 6. Verificación Post-Deploy

#### Health Checks
- ✅ `https://tu-app.railway.app/health/`
- ✅ `https://tu-app.railway.app/admin/`

#### Logs Importantes
```bash
✅ Migraciones ejecutadas
✅ Archivos estáticos recopilados  
✅ Superusuario creado
✅ Servidor iniciado en puerto $PORT
```

### 7. Configuración de Dominio

1. **Settings** → **Networking**
2. **Custom Domain** (opcional)
3. **Generate Domain** para obtener subdominio Railway

### 8. Troubleshooting

#### Problemas Comunes
- **Build fails**: Verificar Dockerfile path
- **Health check fails**: Verificar endpoint `/health/`
- **Database errors**: Verificar `DATABASE_URL`
- **Static files**: Verificar `collectstatic` en build

#### Comandos de Debug
- Ver logs: **Deployments** → **View Logs**
- Restart: **Settings** → **Restart**
- Variables: **Variables** tab

### 9. Estructura Final

```
Railway Project: farmacia-ia
├── farmacia-postgres (PostgreSQL)
├── farmacia-redis (Redis)  
└── farmacia-django (Django App)
    ├── Variables: 15+ configuradas
    ├── Health Check: /health/
    └── Domain: xxx.railway.app
```

### 10. URLs Finales

- **App**: `https://farmacia-django-production.railway.app`
- **Admin**: `https://farmacia-django-production.railway.app/admin`
- **Health**: `https://farmacia-django-production.railway.app/health/`
- **API**: `https://farmacia-django-production.railway.app/conversations/api/`

## 🎉 ¡Listo para usar!

Usuario admin creado: `admin` / `admin123`
