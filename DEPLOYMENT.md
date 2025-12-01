# Deployment Guide for Render

This guide will help you deploy the Stock Movement Direction Predictor to Render.

## Prerequisites

- A [Render account](https://render.com/) (free tier available)
- Your code pushed to a GitHub repository

## Deployment Steps

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit - Stock Predictor App"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

### 2. Create Web Service on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:

   - **Name**: `stock-predictor` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd web && gunicorn app:app`
   - **Instance Type**: `Free` (or choose paid for better performance)

### 3. Environment Variables (Optional)

If you need to set any environment variables:
- Click **"Advanced"**
- Add environment variables as needed

### 4. Deploy

1. Click **"Create Web Service"**
2. Render will automatically:
   - Install dependencies from `requirements.txt`
   - Start the application using gunicorn
   - Provide you with a public URL

### 5. Access Your App

Once deployed, your app will be available at:
```
https://your-app-name.onrender.com
```

## Alternative: Using render.yaml

You can also use the included `render.yaml` file for automatic deployment:

1. In your Render dashboard, click **"New +"** → **"Blueprint"**
2. Connect your repository
3. Render will automatically detect `render.yaml` and configure everything

## Troubleshooting

### Build Fails

- Check that all dependencies in `requirements.txt` are compatible
- Verify Python version (3.10+ recommended)

### App Doesn't Start

- Check the logs in Render dashboard
- Ensure gunicorn is installed (`pip install gunicorn`)
- Verify the start command: `cd web && gunicorn app:app`

### Slow Performance on Free Tier

- Free tier instances spin down after inactivity
- First request after inactivity may take 30-60 seconds
- Consider upgrading to a paid tier for production use

## Production Optimizations

For better performance in production:

1. **Use a paid tier** for always-on instances
2. **Add caching** with Redis for model results
3. **Use a database** instead of in-memory cache
4. **Enable auto-scaling** for high traffic

## Support

For issues or questions:
- Check Render documentation: https://render.com/docs
- Review application logs in Render dashboard
- Open an issue on GitHub

---

**Note**: The free tier on Render is great for testing and demos, but for production use with consistent traffic, consider upgrading to a paid plan.
