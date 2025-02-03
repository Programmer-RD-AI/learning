import asyncio
from playwright.async_api import async_playwright


async def convert_webpage_to_pdf(url, output_pdf):
    async with async_playwright() as p:
        # Launch a headless Chromium instance.
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to the URL and wait until the network is idle.
        await page.goto(url, wait_until="networkidle")

        # Generate the PDF.
        await page.pdf(path=output_pdf, format="A4", print_background=True)

        # Close the browser.
        await browser.close()


if __name__ == "__main__":
    target_url = "https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-the-forking-policy-for-your-repository"
    output_pdf = "output.pdf"
    asyncio.run(convert_webpage_to_pdf(target_url, output_pdf))
