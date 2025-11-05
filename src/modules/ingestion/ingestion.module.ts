/* eslint-disable prettier/prettier */
  import { Module } from '@nestjs/common';              
  import { MulterModule } from '@nestjs/platform-express';                                                                                        
  import { IngestionController } from './ingestion.controller';                                                                                   
  import { IngestionService } from './ingestion.service';                                                                                         
  import { EmbeddingModule } from '../embedding/embedding.module';                                                                                
  import { VectorModule } from '../vector/vector.module';                                                                                         
                                                                                                                                                  
  @Module({                                                                                                                                       
    imports: [MulterModule.register(), EmbeddingModule, VectorModule],
    controllers: [IngestionController],                                                                                                           
    providers: [IngestionService],
    exports: [IngestionService], // ← makes it available to other modules                                                                         
  })                                                                                                                                              
  export class IngestionModule {}