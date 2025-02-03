import asyncio


async def convert_webpage_to_pdf(url, output_path):
    from pyppeteer import launch

    # Launch a headless instance of Chrome
    browser = await launch(headless=True, args=["--no-sandbox"])
    page = await browser.newPage()

    # Navigate to the URL.
    # Using 'networkidle2' waits until network activity subsides,
    # helping to ensure the page (including dynamic content) is fully loaded.
    await page.goto(url, {"waitUntil": "networkidle2"})

    # Generate PDF with settings:
    # - 'path': output file name.
    # - 'format': paper size (e.g., A4).
    # - 'printBackground': include CSS background images/colors.
    await page.pdf({"path": output_path, "format": "A4", "printBackground": True})

    # Close the browser to free resources
    await browser.close()


if __name__ == "__main__":
    # Replace with your target URL and desired output filename.
    target_url = "https://interviewready.io/blog/system-design-of-whatsapp-calling-app"
    output_pdf = "output.pdf"
    asyncio.get_event_loop().run_until_complete(
        convert_webpage_to_pdf(target_url, output_pdf)
    )

# import asyncio
# from pyppeteer.launcher import launch


# async def main():
#     browser = await launch()
#     page = await browser.newPage()
#     await page.goto(
#         "https://interviewready.io/blog/system-design-of-whatsapp-calling-app"
#     )
#     await page.screenshot({"path": "example.png"})
#     await browser.close()


# asyncio.get_event_loop().run_until_complete(main())
