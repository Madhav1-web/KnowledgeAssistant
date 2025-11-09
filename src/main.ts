import 'dotenv/config';
import { NestFactory } from '@nestjs/core';
import { NestExpressApplication } from '@nestjs/platform-express';
import { join } from 'path';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule);
  app.useStaticAssets(join(__dirname, '..', 'public'));
  const port = process.env.PORT ?? 3000;
  try {
    await app.listen(port);
  } catch (err: any) {
    if (err.code === 'EADDRINUSE') {
      console.log(`Port ${port} in use, retrying in 1s...`);
      await new Promise(r => setTimeout(r, 1000));
      await app.listen(port);
    } else {
      throw err;
    }
  }
}
bootstrap();
