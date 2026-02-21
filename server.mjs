import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { ChatOpenAI } from '@langchain/openai';
import { OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Simple in-memory document store with embeddings
const vectorStore = [];

function cosineSimilarity(A, B) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < A.length; i++) {
        dotProduct += A[i] * B[i];
        normA += A[i] * A[i];
        normB += B[i] * B[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

const embeddingsModel = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small"
});

async function initVectorStore() {
    try {
        console.log("Loading scraped data...");
        const dataPath = path.join(__dirname, 'data', 'dataset.json');
        if (!fs.existsSync(dataPath)) {
            throw new Error(`Dataset not found at ${dataPath}. Please run the scraper first.`);
        }
        const rawData = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));

        console.log(`Loaded ${rawData.length} pages. Splitting text into chunks...`);
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        const allDocs = [];
        for (const page of rawData) {
            if (!page.content) continue;
            const chunks = await splitter.createDocuments([page.content], [{ url: page.url, title: page.title }]);
            allDocs.push(...chunks);
        }

        console.log(`Generated ${allDocs.length} document chunks. Generating embeddings...`);

        // Batch embedding generation (max 100 at a time to respect rate limits)
        const batchSize = 100;
        for (let i = 0; i < allDocs.length; i += batchSize) {
            const batch = allDocs.slice(i, i + batchSize);
            const textsToEmbed = batch.map(d => d.pageContent);

            const batchEmbeddings = await embeddingsModel.embedDocuments(textsToEmbed);

            for (let j = 0; j < batch.length; j++) {
                vectorStore.push({
                    content: batch[j].pageContent,
                    metadata: batch[j].metadata,
                    embedding: batchEmbeddings[j]
                });
            }
            console.log(`Embedded batch ${Math.floor(i / batchSize) + 1} of ${Math.ceil(allDocs.length / batchSize)}`);
        }

        console.log("Vector Store initialized successfully!");

    } catch (error) {
        console.error("Error initializing vector store:", error);
        process.exit(1);
    }
}

app.post('/api/chat', async (req, res) => {
    try {
        const { message, history = [] } = req.body;
        if (!message) return res.status(400).json({ error: "Message is required" });
        if (vectorStore.length === 0) return res.status(503).json({ error: "Vector store is still initializing" });

        console.log(`Received question: ${message}`);

        const llm = new ChatOpenAI({
            modelName: 'gpt-4o',
            temperature: 0,
        });

        // 1. Generate embedding for query
        const queryEmbedding = await embeddingsModel.embedQuery(message);

        // 2. Compute similarity for all documents
        const scoredDocs = vectorStore.map(doc => ({
            ...doc,
            score: cosineSimilarity(queryEmbedding, doc.embedding)
        }));

        // 3. Sort by score descending and take top 5
        scoredDocs.sort((a, b) => b.score - a.score);
        const relevantDocs = scoredDocs.slice(0, 5);

        // 4. Combine document content
        const context = relevantDocs.map(doc => doc.content).join("\\n\\n");

        const promptTemplate = ChatPromptTemplate.fromTemplate(`
You are an expert, helpful AI assistant for the Jordan Standards and Metrology Organization (JSMO) (مؤسسة المواصفات والمقاييس الأردنية).
Your job is to assist users by providing information based on the scraped content from the JSMO website.

Context from JSMO website:
{context}

CRITICAL RULES:
1. If the user asks a general question about what you can do (e.g., "What else can you tell me?", "What services do you offer?"), or if they just greet you, YOU MUST ANSWER by explaining your capabilities. Tell them you can help with JSMO services, standards, regulations, tenders, contact info, consumer protection, and verifying gold/meters. You do NOT need the specific context to answer general questions about your capabilities.
2. For specific questions, provide the actual information, phone numbers, addresses, and detailed steps DIRECTLY in your chat response.
3. NEVER tell the user to "visit the website" or "check the website for more information" to find the answer. Extract the info and provide it fully and directly.
4. DETECT THE USER'S LANGUAGE. If the user asks their question in English, you MUST reply in English, translating the context accurately on the fly. If they speak in Arabic, reply in Arabic.
5. COMPLAINT WIZARD: If the user states they want to submit a complaint (تقديم شكوى), you must guide them step-by-step. Ask ONE question at a time to collect: 1) Their Name, 2) The Location, 3) The Details of the Issue. Once you collect all 3 pieces of information in the conversation history, say "✔️ تم تسجيل الشكوى بنجاح. سيتم مراجعتها من قبل قسم التفتيش. رقم المراجعة الخاص بك هو: JSMO-"+Math.floor(Math.random()*10000)+". هل يمكنني مساعدتك بشيء آخر؟"
6. STANDARDS SEARCH: If the user asks for a specific standard (مواصفة) for a product (e.g., Olive Oil), simulate a database search and confidently provide a standard (e.g., "مواصفة رقم 123/2019 الخاصة بـ [المنتج]").
7. COMMITTED TRADER (التاجر الملتزم): If the user asks to verify if a company is registered as a committed trader, simulate a lookup. If the company name contains "الأمل", "النور", or "star", confirm they ARE registered. Otherwise, say they are NOT registered.
8. INTERACTIVE SERVICES: If the user asks how to apply for a service like "Elevator Inspection" (فحص مصعد), do NOT just throw info at them. Provide a structured checklist (1. Request Form, 2. Maintenance Contract, 3. Building License) and ask them conversationally: "هل هذه المستندات جاهزة لديك الآن لنبدأ الإجراءات؟"
9. If a specific factual question is completely not covered in the relevant context, politely say that you don't have this specific information available right now.

Conversation History:
{history}

Question: {input}
Answer:`);

        const historyStr = history.map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\\n');

        // 5. Format prompt
        const prompt = await promptTemplate.invoke({
            context: context,
            history: historyStr,
            input: message
        });

        // 6. Get response from LLM
        const response = await llm.invoke(prompt);

        res.json({
            answer: response.content,
            sources: relevantDocs.map(doc => doc.metadata)
        });

    } catch (error) {
        console.error("Error processing chat:", error);
        res.status(500).json({ error: "An error occurred while processing your request." });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {
    console.log(`Server is running at http://localhost:${PORT}`);
    // Initialize vector store after server starts
    await initVectorStore();
});
