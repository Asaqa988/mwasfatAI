const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');
const path = require('path');
const pdfParse = require('pdf-parse');

const MAX_PAGES = 50;
const BASE_URL = 'https://www.jsmo.gov.jo';
const START_URL = 'https://www.jsmo.gov.jo/Default/Ar/';
const visited = new Set();
const queue = [START_URL];
const results = [];

async function crawl() {
    console.log(`Starting crawl...`);

    // Ensure data directory exists
    const dataDir = path.join(__dirname, 'data');
    if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir);
    }

    while (queue.length > 0 && visited.size < MAX_PAGES) {
        const url = queue.shift();

        if (visited.has(url)) continue;
        visited.add(url);

        console.log(`[${visited.size}/${MAX_PAGES}] Fetching: ${url}`);

        try {
            if (url.toLowerCase().endsWith('.pdf')) {
                console.log(`Downloading and parsing PDF: ${url}`);
                const response = await axios.get(url, { responseType: 'arraybuffer', timeout: 30000 });
                try {
                    const pdfData = await pdfParse(response.data);
                    results.push({
                        url,
                        title: url.split('/').pop() || 'PDF Document',
                        content: pdfData.text.replace(/\s+/g, ' ').trim()
                    });
                } catch (err) {
                    console.error(`Failed to parse PDF ${url}:`, err.message);
                }
                await new Promise(r => setTimeout(r, 500));
                continue;
            }

            // JSMO website might require User-Agent to prevent 403s
            const response = await axios.get(url, {
                headers: {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                },
                timeout: 10000
            });

            const html = response.data;
            const $ = cheerio.load(html);

            const title = $('title').text().trim();

            // Remove scripts, styles, boilerplate navigation, headers, footers
            $('script, style, noscript, iframe, img, svg, video, audio, link, meta, header, footer, nav, .header, .footer, .navbar, .menu, #menu, .sidebar, aside').remove();

            // JSMO uses SharePoint. The main content is usually inside #DeltaPlaceHolderMain or .ms-rtestate-field
            let mainContent = $('#DeltaPlaceHolderMain, .ms-rtestate-field, .article-content, #contentRow, .MainContent');

            let textContent;
            if (mainContent.length > 0) {
                // Remove the accessibility block explicitly if it's inside
                mainContent.find('.accessibility-widget, #accessibility, [class*="accessibility"]').remove();
                textContent = mainContent.text().replace(/\s+/g, ' ').trim();
            } else {
                // Fallback: get body but remove known boilerplate text
                const bodyText = $('body').text().replace(/\s+/g, ' ').trim();
                textContent = bodyText.replace(/بحاجة الى مساعدة\؟ لتعديل موقع الويب وفقًا لاحتياجات الوصول الخاصة بك، حدد خيارًا واحدًا أو أكثر أدناه.*?اعادة الضبط/g, '');
                textContent = textContent.replace(/منصة تصفح بأمان البحث في الموقع English.*?اتصل بنا الرئيسية العطاءات طلب الحصول على المعلومة الأسئلة الأكثر تكرارا خريطة الموقع/g, '');
            }

            // Clean up common leftovers
            textContent = textContent.replace(/شارك شارك/g, '').trim();
            if (textContent.length < 50) {
                textContent = $('body').text().replace(/\s+/g, ' ').trim(); // Ultimate fallback
            }

            results.push({
                url,
                title,
                content: textContent
            });

            // Find more links
            $('a').each((i, link) => {
                let href = $(link).attr('href');
                if (!href) return;

                // Handle relative URLs
                if (href.startsWith('/')) {
                    href = BASE_URL + href;
                }

                // Only enqueue JSMO links that we haven't visited and are not in queue
                if (href.startsWith(BASE_URL) && !visited.has(href) && !queue.includes(href)) {
                    // Ignore files like docx, etc for now to keep it simple, but ALLOW pdfs
                    if (!href.match(/\.(doc|docx|xls|xlsx|zip|rar)$/i)) {
                        queue.push(href);
                    }
                }
            });

        } catch (error) {
            console.error(`Error fetching ${url}:`, error.message);
        }

        // Small delay to be polite
        await new Promise(r => setTimeout(r, 500));
    }

    console.log(`Finished crawling ${results.length} pages.`);

    fs.writeFileSync(
        path.join(dataDir, 'dataset.json'),
        JSON.stringify(results, null, 2)
    );
    console.log('Saved to data/dataset.json');
}

crawl().catch(console.error);
