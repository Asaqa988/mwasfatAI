require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { ChatOpenAI } = require('@langchain/openai');
const { OpenAIEmbeddings } = require('@langchain/openai');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { Document } = require('@langchain/core/documents');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { createRetrievalChain } = require('langchain/chains/retrieval');
const { createStuffDocumentsChain } = require('langchain/chains/combine_documents');
const { ChatPromptTemplate } = require('@langchain/core/prompts');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

let vectorStore = null;

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

        const docs = [];
        for (const page of rawData) {
            if (!page.content) continue;
            const chunks = await splitter.createDocuments([page.content], [{ url: page.url, title: page.title }]);
            docs.push(...chunks);
        }

        console.log(`Generated ${docs.length} document chunks. Initializing MemoryVectorStore...`);
        vectorStore = await MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());
        console.log("Vector Store initialized successfully!");

    } catch (error) {
        console.error("Error initializing vector store:", error);
        process.exit(1);
    }
}

app.post('/api/chat', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: "Message is required" });
        if (!vectorStore) return res.status(503).json({ error: "Vector store is still initializing" });

        console.log(`Received question: ${message}`);

        const llm = new ChatOpenAI({
            modelName: 'gpt-4o',
            temperature: 0,
        });

        const promptTemplate = ChatPromptTemplate.fromTemplate(`
You are an expert AI assistant for the Jordan Standards and Metrology Organization (JSMO).
Use the following pieces of retrieved context from the JSMO website to answer the user's question accurately in Arabic unless requested otherwise.
If the answer is not contained in the context, politely say that you don't know based on the available information from the website.

Context: {context}

Question: {input}
Answer:`);

        const combineDocsChain = await createStuffDocumentsChain({
            llm,
            prompt: promptTemplate,
        });

        const retriever = vectorStore.asRetriever({
            k: 5 // Retrieve top 5 most relevant chunks
        });

        const retrievalChain = await createRetrievalChain({
            retriever,
            combineDocsChain,
        });

        const response = await retrievalChain.invoke({
            input: message,
        });

        res.json({
            answer: response.answer,
            sources: response.context.map(doc => ({ url: doc.metadata.url, title: doc.metadata.title }))
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
