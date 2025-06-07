import os
import tempfile
from typing import List

import nest_asyncio
import requests
import streamlit as st
from agentic_rag import get_agentic_rag_agent
from agno.agent import Agent
from agno.document import Document
from agno.document.reader.csv_reader import CSVReader
from agno.document.reader.pdf_reader import PDFReader
from agno.document.reader.text_reader import TextReader
from agno.document.reader.website_reader import WebsiteReader
from agno.utils.log import logger
from utils import (
    CUSTOM_CSS,
    about_widget,
    add_message,
    display_tool_calls,
    export_chat_history,
    rename_session_widget,
    session_selector_widget,
)


import base64
import json
import os
import re
import time
import tempfile
from typing import Union, List, IO, Any
from pathlib import Path
import concurrent.futures
from datetime import datetime
from openai import OpenAI
from agno.document.base import Document
from agno.document.reader.base import  Reader as BasePDFImageReader
from agno.utils.log import log_info, logger


api_key = st.secrets["env"]["API_KEY"]



client = OpenAI(api_key=api_key)

class PDFImageReader(BasePDFImageReader):
    """Reader for PDF files with OpenAI Vision captioning of images"""

    def caption_image(self, image_bytes: bytes, context_text: str = "") -> str:
        try:
            prompt = "Describe what's in this image with extraction of any visible text."
            # if context_text:
                # prompt += " Consider this potential context for the image: " + context_text
                
            logger.info("Sending image for captioning with context...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
                            }}
                        ]
                    }
                ],
                max_tokens=1000,
            )
            caption = response.choices[0].message.content.strip()
            logger.info(f"Caption received: {caption}")
            return caption
        except Exception as e:
            logger.warning(f"Failed to caption image: {e}")
            return ""
            
    def caption_images_batch(self, image_data_list, context_texts=None, max_workers=5):
        """Process multiple images in parallel using ThreadPoolExecutor with context"""
        logger.info(f"Processing batch of {len(image_data_list)} images in parallel")
        captions = []
        
        if context_texts is None:
            context_texts = [""] * len(image_data_list)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all caption tasks with context
            future_to_image = {
                executor.submit(self.caption_image, image_data, context_texts[i]): i 
                for i, image_data in enumerate(image_data_list)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_image):
                image_idx = future_to_image[future]
                try:
                    caption = future.result()
                    captions.append((image_idx, caption))
                except Exception as e:
                    logger.warning(f"Exception processing image {image_idx}: {e}")
                    captions.append((image_idx, ""))
                    
        # Sort captions by original index and return just the captions
        return [caption for _, caption in sorted(captions)]

    def create_batch_requests_for_all_images(self, all_image_data):
        """Create batch request tasks for all images in the PDF"""
        tasks = []
        for image_info in all_image_data:
            image_b64 = base64.b64encode(image_info['data']).decode()
            prompt = "Describe what's in this image with extraction of any visible text."
            if image_info['context']:
                prompt += f" Consider this potential context for the image: {image_info['context']}"
                
            task = {
                "custom_id": f"page_{image_info['page']}_image_{image_info['image_idx']}",
                "method": "POST", 
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }}
                            ]
                        }
                    ],
                    "max_tokens": 1000
                }
            }
            tasks.append(task)
        return tasks
    
    def filter_unwanted_images(self, images, page_number):
        """
        Filter out unwanted images like placeholders, logos, very small images, etc.
        """
        filtered_images = []
        
        for idx, img in enumerate(images):
            try:
                # Get image data size
                image_size = len(img.data)
                
                # Filter 1: Remove very small images (likely logos, icons, placeholders)
                if image_size < 5000:  # Less than 5KB
                    logger.debug(f"Filtered small image on page {page_number}: {image_size} bytes")
                    continue
                
                # Filter 2: Try to get image dimensions if possible
                try:
                    from PIL import Image
                    import io
                    
                    # Try to open image to get dimensions
                    pil_image = Image.open(io.BytesIO(img.data))
                    width, height = pil_image.size
                    
                    # Filter very small dimensions (likely icons/logos)
                    if width < 50 or height < 50:
                        logger.debug(f"Filtered tiny image on page {page_number}: {width}x{height}")
                        continue
                    
                    # Filter very thin images (likely decorative lines/borders)
                    if (width < 20 and height > 100) or (height < 20 and width > 100):
                        logger.debug(f"Filtered thin decorative image on page {page_number}: {width}x{height}")
                        continue
                    
                    # Filter very small square images (likely logos/icons)
                    if width <= 100 and height <= 100:
                        logger.debug(f"Filtered small square image on page {page_number}: {width}x{height}")
                        continue
                    
                    # Filter images with very low aspect ratios that are likely backgrounds/decorations
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > 10:  # Very wide or very tall images
                        logger.debug(f"Filtered extreme aspect ratio image on page {page_number}: {width}x{height} (ratio: {aspect_ratio:.2f})")
                        continue
                    
                    logger.debug(f"Keeping image on page {page_number}: {width}x{height}, {image_size} bytes")
                    
                except ImportError:
                    logger.warning("PIL not available, using size-based filtering only")
                    # If PIL is not available, only use size-based filtering
                    pass
                except Exception as e:
                    logger.debug(f"Could not analyze image dimensions on page {page_number}: {e}")
                    # If we can't analyze the image, keep it if it's reasonably sized
                    pass
                
                # Filter 3: Remove extremely large images that might be full-page backgrounds
                if image_size > 2_000_000:  # Larger than 2MB
                    logger.debug(f"Filtered very large image on page {page_number}: {image_size} bytes")
                    continue
                
                # If image passes all filters, keep it
                filtered_images.append(img)
                
            except Exception as e:
                logger.warning(f"Error filtering image {idx} on page {page_number}: {e}")
                # If there's an error, err on the side of keeping the image
                filtered_images.append(img)
        
        return filtered_images

    def process_all_images_with_batch_api(self, all_image_data, batch_size=100, timeout=3600):
        """Process all PDF images using OpenAI Batch API in larger batches"""
        if not all_image_data:
            return {}, []
            
        total_images = len(all_image_data)
        logger.info(f"Processing {total_images} total images using Batch API in batches of {batch_size}")
        
        all_results = {}
        batch_reports = []
        
        # Process images in chunks of batch_size
        for batch_start in range(0, total_images, batch_size):
            batch_start_time = time.time()
            batch_end = min(batch_start + batch_size, total_images)
            current_batch = all_image_data[batch_start:batch_end]
            current_batch_size = len(current_batch)
            
            logger.info(f"Creating batch job for images {batch_start+1}-{batch_end} ({current_batch_size} images)")
            
            # Create batch requests for current chunk
            tasks = self.create_batch_requests_for_all_images(current_batch)
            
            # Save to temporary JSONL file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for task in tasks:
                    f.write(json.dumps(task) + '\n')
                batch_file_path = f.name
            
            batch_report = {
                "batch_number": (batch_start // batch_size) + 1,
                "image_range": f"{batch_start+1}-{batch_end}",
                "image_count": current_batch_size,
                "start_time": datetime.fromtimestamp(batch_start_time).isoformat(),
                "status": "started"
            }
            
            try:
                # Upload batch file
                upload_start = time.time()
                batch_file = client.files.create(
                    file=open(batch_file_path, "rb"),
                    purpose="batch"
                )
                upload_time = time.time() - upload_start
                
                # Create batch job
                job_create_start = time.time()
                batch_job = client.batches.create(
                    input_file_id=batch_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": f"PDF_images_batch_{batch_start+1}_{batch_end}",
                        "batch_range": f"{batch_start+1}-{batch_end}",
                        "image_count": str(current_batch_size)
                    }
                )
                job_create_time = time.time() - job_create_start
                
                batch_report.update({
                    "batch_id": batch_job.id,
                    "upload_time_seconds": round(upload_time, 2),
                    "job_create_time_seconds": round(job_create_time, 2)
                })
                
                logger.info(f"Batch job created: {batch_job.id} for images {batch_start+1}-{batch_end}")
                
                # Wait for completion
                processing_start = time.time()
                batch_results = self._wait_for_batch_completion(batch_job.id, f"images {batch_start+1}-{batch_end}", timeout)
                processing_time = time.time() - processing_start
                
                # Add results to main results dict
                successful_results = 0
                failed_results = 0
                for custom_id, caption in batch_results:
                    all_results[custom_id] = caption
                    if caption.strip():
                        successful_results += 1
                    else:
                        failed_results += 1
                
                # Calculate costs
                estimated_input_tokens = current_batch_size * 1500
                estimated_output_tokens = current_batch_size * 300
                estimated_cost = (estimated_input_tokens / 1_000_000) * 0.075 + (estimated_output_tokens / 1_000_000) * 0.30
                
                batch_end_time = time.time()
                total_batch_time = batch_end_time - batch_start_time
                
                batch_report.update({
                    "status": "completed",
                    "processing_time_seconds": round(processing_time, 2),
                    "total_time_seconds": round(total_batch_time, 2),
                    "successful_results": successful_results,
                    "failed_results": failed_results,
                    "estimated_input_tokens": estimated_input_tokens,
                    "estimated_output_tokens": estimated_output_tokens,
                    "estimated_cost_usd": round(estimated_cost, 4),
                    "end_time": datetime.fromtimestamp(batch_end_time).isoformat()
                })
                
                # Log cost estimation for this batch
                self._log_batch_cost_estimate(batch_job.id, current_batch_size)
                
            except Exception as e:
                batch_end_time = time.time()
                total_batch_time = batch_end_time - batch_start_time
                
                batch_report.update({
                    "status": "failed",
                    "error": str(e),
                    "total_time_seconds": round(total_batch_time, 2),
                    "end_time": datetime.fromtimestamp(batch_end_time).isoformat()
                })
                logger.error(f"Batch processing failed: {e}")
                
            finally:
                # Cleanup temp file
                Path(batch_file_path).unlink(missing_ok=True)
                batch_reports.append(batch_report)
        
        logger.info(f"Completed processing all {total_images} images across {(total_images + batch_size - 1) // batch_size} batch jobs")
        return all_results, batch_reports
    
    def _wait_for_batch_completion(self, batch_id, description, timeout):
        """Wait for batch completion with polling"""
        logger.info(f"Waiting for batch {batch_id} ({description}) to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                batch = client.batches.retrieve(batch_id)
                
                if batch.status == "completed":
                    logger.info(f"Batch {batch_id} completed successfully")
                    return self._download_batch_results(batch.output_file_id)
                elif batch.status == "failed":
                    logger.error(f"Batch {batch_id} failed")
                    if hasattr(batch, 'errors') and batch.errors:
                        logger.error(f"Batch errors: {batch.errors}")
                    return []
                elif batch.status in ["validating", "in_progress", "finalizing"]:
                    elapsed = int(time.time() - start_time)
                    logger.info(f"Batch {batch_id} status: {batch.status} (elapsed: {elapsed}s)")
                    time.sleep(30)  # Wait 30 seconds before next check
                else:
                    logger.warning(f"Unexpected batch status: {batch.status}")
                    time.sleep(30)
                    
            except Exception as e:
                logger.error(f"Error checking batch status: {e}")
                time.sleep(30)
        
        logger.warning(f"Batch {batch_id} timed out after {timeout} seconds")
        return []
    
    def _download_batch_results(self, output_file_id):
        """Download and parse batch results"""
        if not output_file_id:
            logger.warning("No output file ID provided")
            return []
        
        try:
            # Download results file
            result_file = client.files.content(output_file_id)
            results = []
            
            for line in result_file.text.strip().split('\n'):
                if not line.strip():
                    continue
                    
                try:
                    result = json.loads(line)
                    custom_id = result.get('custom_id', '')
                    
                    if result.get('response', {}).get('status_code') == 200:
                        caption = result['response']['body']['choices'][0]['message']['content'].strip()
                        results.append((custom_id, caption))
                        logger.debug(f"Successfully processed {custom_id}")
                    else:
                        error_msg = result.get('response', {}).get('body', {}).get('error', {}).get('message', 'Unknown error')
                        logger.warning(f"Failed request {custom_id}: {error_msg}")
                        results.append((custom_id, ""))
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse result line: {e}")
                    continue
            
            logger.info(f"Downloaded {len(results)} batch results")
            return results
            
        except Exception as e:
            logger.error(f"Error downloading batch results: {e}")
            return []

    def _log_batch_cost_estimate(self, batch_id, image_count):
        """Log estimated cost for the batch job"""
        try:
            batch_obj = client.batches.retrieve(batch_id)
            
            # Get request counts if available
            total_requests = image_count  # Fallback to image count
            if hasattr(batch_obj, 'request_counts') and batch_obj.request_counts:
                total_requests = getattr(batch_obj.request_counts, 'total', image_count)
                completed_requests = getattr(batch_obj.request_counts, 'completed', 0)
                failed_requests = getattr(batch_obj.request_counts, 'failed', 0)
                
                logger.info(f"Batch {batch_id} summary:")
                logger.info(f"  Total requests: {total_requests}")
                logger.info(f"  Completed: {completed_requests}")
                logger.info(f"  Failed: {failed_requests}")
            
            # Cost estimation
            estimated_input_tokens_per_image = 1500  # Image + context + prompt
            estimated_output_tokens_per_image = 300   # Caption
            
            total_input_tokens = total_requests * estimated_input_tokens_per_image
            total_output_tokens = total_requests * estimated_output_tokens_per_image
            
            # Batch API pricing (50% discount)
            input_cost = (total_input_tokens / 1_000_000) * 0.075
            output_cost = (total_output_tokens / 1_000_000) * 0.30
            total_batch_cost = input_cost + output_cost
            
            # Real-time API pricing (for comparison)
            realtime_input_cost = (total_input_tokens / 1_000_000) * 0.15
            realtime_output_cost = (total_output_tokens / 1_000_000) * 0.60
            total_realtime_cost = realtime_input_cost + realtime_output_cost
            
            savings = total_realtime_cost - total_batch_cost
            
            logger.info(f"💰 Cost Estimation for Batch {batch_id}:")
            logger.info(f"  📊 Estimated input tokens: {total_input_tokens:,}")
            logger.info(f"  📊 Estimated output tokens: {total_output_tokens:,}")
            logger.info(f"  💵 Batch API cost: ${total_batch_cost:.4f}")
            logger.info(f"  💵 Real-time API cost: ${total_realtime_cost:.4f}")
            logger.info(f"  💰 Savings: ${savings:.4f} ({(savings/total_realtime_cost*100):.1f}%)")
                
        except Exception as e:
            logger.warning(f"Could not retrieve cost information for batch {batch_id}: {e}")
    
    def extract_context_for_image(self, page_texts, current_page_idx, line_window=10, extra_context_window=10):
        """
        Extract text context for an image by including surrounding text lines,
        considering text from previous, current, and next pages.
        - `line_window`: base number of lines before and after.
        - `extra_context_window`: additional lines from prev/next pages for headings or captions.
        """
        def get_lines(text):
            return text.split('\n') if text else []

        context_lines = []

        # Previous page context
        if current_page_idx > 0:
            prev_lines = get_lines(page_texts[current_page_idx - 1])
            context_lines += prev_lines[-(line_window + extra_context_window):]

        # Current page context
        current_lines = get_lines(page_texts[current_page_idx])
        context_lines += current_lines

        # Next page context
        if current_page_idx < len(page_texts) - 1:
            next_lines = get_lines(page_texts[current_page_idx + 1])
            context_lines += next_lines[:(line_window + extra_context_window)]

        # Combine and truncate if overly long
        context = '\n'.join(context_lines)

        # Attempt to extract potential headings or figure/table captions
        caption_patterns = [
            r"(?:Figure|Fig\.?|Table)\s*\d+[.:]\s*[^\n]+",  # Figure/Table captions
            r"(?:[A-Z][a-z]+\s*){1,5}:",                    # Potential headings with colon
            r"\d+\.\d+\s+[A-Z][a-zA-Z\s]+",                 # Numbered section headings
        ]

        extracted_captions = []
        for pattern in caption_patterns:
            matches = re.findall(pattern, context)
            extracted_captions.extend(matches)

        if extracted_captions:
            context += "\nPotential captions/headings: " + "; ".join(set(extracted_captions))

        return context

    def read(self, pdf: Union[str, Path, IO[Any]], output_json_path=None, use_batch_api=True, batch_size=100) -> List[Document]:
        """
        Read PDF with image captioning
        
        Args:
            pdf: PDF file path or file object
            output_json_path: Path to save extraction data JSON
            use_batch_api: If True, use OpenAI Batch API for 50% cost savings (slower processing)
            batch_size: Number of images to process in each batch job (default: 100)
        """
        if not pdf:
            raise ValueError("No pdf provided")

        try:
            from pypdf import PdfReader as DocumentReader
        except ImportError:
            raise ImportError("`pypdf` not installed")

        try:
            if isinstance(pdf, str):
                doc_name = pdf.split("/")[-1].split(".")[0].replace(" ", "_")
                # Use the PDF filename to create a default JSON filename if not provided
                if output_json_path is None:
                    output_dir = os.path.dirname(pdf) or "."
                    output_json_path = os.path.join(output_dir, f"{doc_name}_extraction.json")
            else:
                doc_name = pdf.name.split(".")[0]
                if output_json_path is None:
                    output_json_path = f"{doc_name}_extraction.json"
        except Exception:
            doc_name = "pdf"
            if output_json_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_json_path = f"pdf_extraction_{timestamp}.json"

        logger.info(f"Reading PDF: {doc_name}")
        logger.info(f"🚀 Using {'Batch API (50% cheaper)' if use_batch_api else 'Real-time API'} for image processing")
        if use_batch_api:
            logger.info(f"📦 Batch size: {batch_size} images per batch job")
        
        doc_reader = DocumentReader(pdf)
        
        # First pass: extract all page texts for context building
        all_page_texts = []
        for page in doc_reader.pages:
            page_text = page.extract_text() or ""
            all_page_texts.append(page_text)

        # Second pass: collect all images from all pages with filtering
        all_image_data = []
        page_image_mapping = {}  # Track which images belong to which page
        
        for page_number, page in enumerate(doc_reader.pages, start=1):
            all_images = list(page.images)
            page_image_mapping[page_number] = []
            
            if all_images:
                logger.info(f"Found {len(all_images)} images on page {page_number}")
                filtered_images = self.filter_unwanted_images(all_images, page_number)
                logger.info(f"After filtering: {len(filtered_images)} images remain on page {page_number}")
                
                for img_idx, img in enumerate(filtered_images):
                    # Extract context for this image
                    context = self.extract_context_for_image(
                        all_page_texts, 
                        page_number-1,  # 0-based index
                        line_window=10
                    )
                    
                    image_info = {
                        'data': img.data,
                        'page': page_number,
                        'image_idx': img_idx,
                        'context': context,
                        'original_size': len(img.data),
                        'filtered': True
                    }
                    
                    all_image_data.append(image_info)
                    page_image_mapping[page_number].append(len(all_image_data) - 1)  # Store global index

        total_images_processed = len(all_image_data)
        logger.info(f"🖼️ Total images found across all pages: {total_images_processed}")

        # Process all images at once
        batch_results = {}
        batch_reports = []
        if total_images_processed > 0:
            if use_batch_api:
                batch_results, batch_reports = self.process_all_images_with_batch_api(all_image_data, batch_size)
            else:
                # Use real-time API with parallel processing
                logger.info("Processing all images using real-time API...")
                realtime_start = time.time()
                
                image_data_list = [img_info['data'] for img_info in all_image_data]
                image_contexts = [img_info['context'] for img_info in all_image_data]
                
                # Process in smaller chunks for real-time API
                realtime_batch_size = 10
                captions = []
                
                for i in range(0, len(image_data_list), realtime_batch_size):
                    batch = image_data_list[i:i + realtime_batch_size]
                    batch_contexts = image_contexts[i:i + realtime_batch_size]
                    batch_captions = self.caption_images_batch(batch, batch_contexts)
                    captions.extend(batch_captions)
                
                realtime_end = time.time()
                realtime_processing_time = realtime_end - realtime_start
                
                # Convert to batch_results format
                successful_results = 0
                for idx, caption in enumerate(captions):
                    img_info = all_image_data[idx]
                    custom_id = f"page_{img_info['page']}_image_{img_info['image_idx']}"
                    batch_results[custom_id] = caption
                    if caption.strip():
                        successful_results += 1
                
                # Create single report for real-time processing
                estimated_input_tokens = total_images_processed * 1500
                estimated_output_tokens = total_images_processed * 300
                estimated_cost = (estimated_input_tokens / 1_000_000) * 0.15 + (estimated_output_tokens / 1_000_000) * 0.60
                
                batch_reports = [{
                    "batch_number": 1,
                    "image_range": f"1-{total_images_processed}",
                    "image_count": total_images_processed,
                    "processing_method": "real_time_api",
                    "start_time": datetime.fromtimestamp(realtime_start).isoformat(),
                    "end_time": datetime.fromtimestamp(realtime_end).isoformat(),
                    "total_time_seconds": round(realtime_processing_time, 2),
                    "successful_results": successful_results,
                    "failed_results": total_images_processed - successful_results,
                    "estimated_input_tokens": estimated_input_tokens,
                    "estimated_output_tokens": estimated_output_tokens,
                    "estimated_cost_usd": round(estimated_cost, 4),
                    "status": "completed"
                }]

        # Third pass: create documents with captions
        documents = []
        extraction_data = {
            "pdf_name": doc_name,
            "extraction_date": datetime.now().isoformat(),
            "total_pages": len(doc_reader.pages),
            "total_images_found": sum(len(list(page.images)) for page in doc_reader.pages),
            "total_images_processed": total_images_processed,
            "images_filtered_out": sum(len(list(page.images)) for page in doc_reader.pages) - total_images_processed,
            "processing_method": "batch_api" if use_batch_api else "real_time_api",
            "batch_size": batch_size if use_batch_api else "N/A",
            "batch_processing_reports": batch_reports,
            "pages": []
        }
        
        for page_number, page in enumerate(doc_reader.pages, start=1):
            logger.info(f"Creating document for page {page_number}")
            page_text = all_page_texts[page_number-1]
            
            page_data = {
                "page_number": page_number,
                "text_content": page_text,
                "images": []
            }
            
            images_text_list = []
            
            # Get images for this page
            if page_number in page_image_mapping:
                for global_img_idx in page_image_mapping[page_number]:
                    img_info = all_image_data[global_img_idx]
                    custom_id = f"page_{page_number}_image_{img_info['image_idx']}"
                    caption = batch_results.get(custom_id, "")
                    
                    if caption:
                        enhanced_caption = f"{caption}\n\nContext: {img_info['context']}"
                        images_text_list.append(enhanced_caption)
                    
                    page_data["images"].append({
                        "image_index": img_info['image_idx'] + 1,
                        "caption": caption,
                        "context_used": img_info['context'],
                        "processing_method": "batch_api" if use_batch_api else "real_time_api",
                        "original_size_bytes": img_info['original_size'],
                        "was_filtered": img_info.get('filtered', False)
                    })
            
            images_text = "\n".join(images_text_list)
            content = page_text + "\n" + images_text

            documents.append(
                Document(
                    name=doc_name,
                    id=f"{doc_name}_{page_number}",
                    meta_data={"page": page_number},
                    content=content,
                )
            )
            
            extraction_data["pages"].append(page_data)

        # Save extraction data to JSON file
        try:
            with open(output_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(extraction_data, json_file, ensure_ascii=False, indent=2)
            logger.info(f"Extraction data saved to {output_json_path}")
        except Exception as e:
            logger.error(f"Failed to save extraction data to JSON: {e}")

        # Log final processing summary with detailed cost breakdown
        total_original_images = sum(len(list(page.images)) for page in doc_reader.pages)
        images_filtered = total_original_images - total_images_processed
        
        logger.info(f"📊 Final Processing Summary:")
        logger.info(f"  📄 Total pages: {len(doc_reader.pages)}")
        logger.info(f"  🖼️  Original images found: {total_original_images}")
        logger.info(f"  🚫 Images filtered out: {images_filtered}")
        logger.info(f"  ✅ Images processed: {total_images_processed}")
        logger.info(f"  ⚡ Method: {'Batch API (50% savings)' if use_batch_api else 'Real-time API'}")
        if use_batch_api and total_images_processed > 0:
            num_batches = (total_images_processed + batch_size - 1) // batch_size
            logger.info(f"  📦 Batch jobs created: {num_batches}")
            logger.info(f"  📦 Images per batch: {batch_size}")
        
        # Add summary to extraction data
        if batch_reports:
            total_processing_time = sum(report.get('total_time_seconds', 0) for report in batch_reports)
            total_estimated_cost = sum(report.get('estimated_cost_usd', 0) for report in batch_reports)
            total_successful = sum(report.get('successful_results', 0) for report in batch_reports)
            total_failed = sum(report.get('failed_results', 0) for report in batch_reports)
            
            extraction_data['processing_summary'] = {
                "total_processing_time_seconds": round(total_processing_time, 2),
                "total_estimated_cost_usd": round(total_estimated_cost, 4),
                "successful_captions": total_successful,
                "failed_captions": total_failed,
                "success_rate_percent": round((total_successful / total_images_processed * 100) if total_images_processed > 0 else 0, 2),
                "average_time_per_image_seconds": round(total_processing_time / total_images_processed, 2) if total_images_processed > 0 else 0,
                "filtering_efficiency": {
                    "original_images": total_original_images,
                    "filtered_out": images_filtered,
                    "processed": total_images_processed,
                    "filter_rate_percent": round((images_filtered / total_original_images * 100) if total_original_images > 0 else 0, 2)
                }
            }
        
        if total_images_processed > 0:
            # Detailed cost calculation for final summary
            estimated_input_tokens_per_image = 1500  # Image + context + prompt
            estimated_output_tokens_per_image = 300   # Caption
            
            total_input_tokens = total_images_processed * estimated_input_tokens_per_image
            total_output_tokens = total_images_processed * estimated_output_tokens_per_image
            
            if use_batch_api:
                # Batch API pricing (50% discount)
                input_cost = (total_input_tokens / 1_000_000) * 0.075
                output_cost = (total_output_tokens / 1_000_000) * 0.30
                final_cost = input_cost + output_cost
                
                # Real-time comparison
                realtime_input_cost = (total_input_tokens / 1_000_000) * 0.15
                realtime_output_cost = (total_output_tokens / 1_000_000) * 0.60
                realtime_cost = realtime_input_cost + realtime_output_cost
                
                savings = realtime_cost - final_cost
                
                logger.info(f"💰 FINAL COST BREAKDOWN:")
                logger.info(f"  📊 Input tokens: {total_input_tokens:,}")
                logger.info(f"  📊 Output tokens: {total_output_tokens:,}")
                logger.info(f"  💵 Batch API Final Cost: ${final_cost:.4f}")
                logger.info(f"  💸 Real-time API would cost: ${realtime_cost:.4f}")
                logger.info(f"  🎉 Total Savings: ${savings:.4f} ({(savings/realtime_cost*100):.1f}%)")
                logger.info(f"  ✨ YOU SAVED ${savings:.4f} by using Batch API!")
            else:
                # Real-time API pricing
                input_cost = (total_input_tokens / 1_000_000) * 0.15
                output_cost = (total_output_tokens / 1_000_000) * 0.60
                final_cost = input_cost + output_cost
                
                # Batch API comparison
                batch_input_cost = (total_input_tokens / 1_000_000) * 0.075
                batch_output_cost = (total_output_tokens / 1_000_000) * 0.30
                batch_cost = batch_input_cost + batch_output_cost
                
                potential_savings = final_cost - batch_cost
                
                logger.info(f"💰 FINAL COST BREAKDOWN:")
                logger.info(f"  📊 Input tokens: {total_input_tokens:,}")
                logger.info(f"  📊 Output tokens: {total_output_tokens:,}")
                logger.info(f"  💵 Real-time API Final Cost: ${final_cost:.4f}")
                logger.info(f"  💡 Batch API would cost: ${batch_cost:.4f}")
                logger.info(f"  💸 Potential Savings: ${potential_savings:.4f} ({(potential_savings/final_cost*100):.1f}%)")
                logger.info(f"  🔄 Next time use use_batch_api=True to save ${potential_savings:.4f}!")
        else:
            logger.info(f"  💵 No images processed - No API costs incurred")

        if self.chunk:
            logger.info("Chunking documents...")
            chunked_documents = []
            for document in documents:
                chunked_documents.extend(self.chunk_document(document))
            logger.info("Chunking completed.")
            return chunked_documents

        logger.info("Finished reading PDF.")
        return documents



nest_asyncio.apply()
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)




def restart_agent():
    """Reset the agent and clear chat history"""
    logger.debug("---*--- Restarting agent ---*---")
    st.session_state["agentic_rag_agent"] = None
    st.session_state["agentic_rag_agent_session_id"] = None
    st.session_state["messages"] = []
    st.rerun()


def get_reader(file_type: str):
    """Return appropriate reader based on file type."""
    readers = {
        "pdf": PDFReader(),
        "csv": CSVReader(),
        "txt": TextReader(),
    }
    return readers.get(file_type.lower(), None)


def main():
    ####################################################################
    # App header
    ####################################################################
    st.markdown("<h1 class='main-title'>Agentic RAG </h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Your intelligent research assistant powered by Agno</p>",
        unsafe_allow_html=True,
    )

    ####################################################################
    # Model selector
    ####################################################################
    model_options = {
        "gpt-4o": "openai:gpt-4o-mini"
    }
    selected_model = st.sidebar.selectbox(
        "Select a model",
        options=list(model_options.keys()),
        index=0,
        key="model_selector",
    )
    model_id = model_options[selected_model]

    ####################################################################
    # Initialize Agent
    ####################################################################
    agentic_rag_agent: Agent
    if (
        "agentic_rag_agent" not in st.session_state
        or st.session_state["agentic_rag_agent"] is None
        or st.session_state.get("current_model") != model_id
    ):
        logger.info("---*--- Creating new Agentic RAG  ---*---")
        agentic_rag_agent = get_agentic_rag_agent(model_id=model_id)
        st.session_state["agentic_rag_agent"] = agentic_rag_agent
        st.session_state["current_model"] = model_id
    else:
        agentic_rag_agent = st.session_state["agentic_rag_agent"]

    ####################################################################
    # Load Agent Session from the database
    ####################################################################
    # Check if session ID is already in session state
    session_id_exists = (
        "agentic_rag_agent_session_id" in st.session_state
        and st.session_state["agentic_rag_agent_session_id"]
    )

    if not session_id_exists:
        try:
            st.session_state["agentic_rag_agent_session_id"] = (
                agentic_rag_agent.load_session()
            )
        except Exception as e:
            logger.error(f"Session load error: {str(e)}")
            st.warning("Could not create Agent session, is the database running?")
            # Continue anyway instead of returning, to avoid breaking session switching
    elif (
        st.session_state["agentic_rag_agent_session_id"]
        and hasattr(agentic_rag_agent, "memory")
        and agentic_rag_agent.memory is not None
        and not agentic_rag_agent.memory.runs
    ):
        # If we have a session ID but no runs, try to load the session explicitly
        try:
            agentic_rag_agent.load_session(
                st.session_state["agentic_rag_agent_session_id"]
            )
        except Exception as e:
            logger.error(f"Failed to load existing session: {str(e)}")
            # Continue anyway

    ####################################################################
    # Load runs from memory
    ####################################################################
    agent_runs = []
    if hasattr(agentic_rag_agent, "memory") and agentic_rag_agent.memory is not None:
        agent_runs = agentic_rag_agent.memory.runs

    # Initialize messages if it doesn't exist yet
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Only populate messages from agent runs if we haven't already
    if len(st.session_state["messages"]) == 0 and len(agent_runs) > 0:
        logger.debug("Loading run history")
        for _run in agent_runs:
            # Check if _run is an object with message attribute
            if hasattr(_run, "message") and _run.message is not None:
                add_message(_run.message.role, _run.message.content)
            # Check if _run is an object with response attribute
            if hasattr(_run, "response") and _run.response is not None:
                add_message("assistant", _run.response.content, _run.response.tools)
    elif len(agent_runs) == 0 and len(st.session_state["messages"]) == 0:
        logger.debug("No run history found")

    if prompt := st.chat_input("👋 Ask me anything!"):
        add_message("user", prompt)

    ####################################################################
    # Track loaded URLs and files in session state
    ####################################################################
    if "loaded_urls" not in st.session_state:
        st.session_state.loaded_urls = set()
    if "loaded_files" not in st.session_state:
        st.session_state.loaded_files = set()
    if "knowledge_base_initialized" not in st.session_state:
        st.session_state.knowledge_base_initialized = False

    st.sidebar.markdown("#### 📚 Document Management")
    input_url = st.sidebar.text_input("Add URL to Knowledge Base")
    if (
        input_url and not prompt and not st.session_state.knowledge_base_initialized
    ):  # Only load if KB not initialized
        if input_url not in st.session_state.loaded_urls:
            alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
            if input_url.lower().endswith(".pdf"):
                try:
                    # Download PDF to temporary file
                    response = requests.get(input_url, stream=True, verify=False)
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                        tmp_path = tmp_file.name

                    reader = PDFReader()
                    docs: List[Document] = reader.read(tmp_path)

                    # Clean up temporary file
                    os.unlink(tmp_path)
                except Exception as e:
                    st.sidebar.error(f"Error processing PDF: {str(e)}")
                    docs = []
            else:
                scraper = WebsiteReader(max_links=2, max_depth=1)
                docs: List[Document] = scraper.read(input_url)

            if docs:
                agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
                st.session_state.loaded_urls.add(input_url)
                st.sidebar.success("URL added to knowledge base")
            else:
                st.sidebar.error("Could not process the provided URL")
            alert.empty()
        else:
            st.sidebar.info("URL already loaded in knowledge base")

    uploaded_file = st.sidebar.file_uploader(
        "Add a Document (.pdf, .csv, or .txt)", key="file_upload"
    )
    if (
        uploaded_file and not prompt and not st.session_state.knowledge_base_initialized
    ):  # Only load if KB not initialized
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_identifier not in st.session_state.loaded_files:
            alert = st.sidebar.info("Processing document...", icon="ℹ️")
            file_type = uploaded_file.name.split(".")[-1].lower()
            reader = get_reader(file_type)
            if reader:
                docs = reader.read(uploaded_file)
                agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
                st.session_state.loaded_files.add(file_identifier)
                st.sidebar.success(f"{uploaded_file.name} added to knowledge base")
                st.session_state.knowledge_base_initialized = True
            alert.empty()
        else:
            st.sidebar.info(f"{uploaded_file.name} already loaded in knowledge base")

    if st.sidebar.button("Clear Knowledge Base"):
        agentic_rag_agent.knowledge.vector_db.delete()
        st.session_state.loaded_urls.clear()
        st.session_state.loaded_files.clear()
        st.session_state.knowledge_base_initialized = False  # Reset initialization flag
        st.sidebar.success("Knowledge base cleared")
    ###############################################################
    # Sample Question
    ###############################################################
    st.sidebar.markdown("#### ❓ Sample Questions")
    if st.sidebar.button("📝 Summarize"):
        add_message(
            "user",
            "Can you summarize what is currently in the knowledge base (use `search_knowledge_base` tool)?",
        )

    ###############################################################
    # Utility buttons
    ###############################################################
    st.sidebar.markdown("#### 🛠️ Utilities")
    col1, col2 = st.sidebar.columns([1, 1])  # Equal width columns
    with col1:
        if st.sidebar.button(
            "🔄 New Chat", use_container_width=True
        ):  # Added use_container_width
            restart_agent()
    with col2:
        if st.sidebar.download_button(
            "💾 Export Chat",
            export_chat_history(),
            file_name="rag_chat_history.md",
            mime="text/markdown",
            use_container_width=True,  # Added use_container_width
        ):
            st.sidebar.success("Chat history exported!")

    ####################################################################
    # Display chat history
    ####################################################################
    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            _content = message["content"]
            if _content is not None:
                with st.chat_message(message["role"]):
                    # Display tool calls if they exist in the message
                    if "tool_calls" in message and message["tool_calls"]:
                        display_tool_calls(st.empty(), message["tool_calls"])
                    st.markdown(_content)

    ####################################################################
    # Generate response for user message
    ####################################################################
    last_message = (
        st.session_state["messages"][-1] if st.session_state["messages"] else None
    )
    if last_message and last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            # Create container for tool calls
            tool_calls_container = st.empty()
            resp_container = st.empty()
            with st.spinner("🤔 Thinking..."):
                response = ""
                try:
                    # Run the agent and stream the response
                    run_response = agentic_rag_agent.run(question, stream=True)
                    for _resp_chunk in run_response:
                        # Display tool calls if available
                        if _resp_chunk.tools and len(_resp_chunk.tools) > 0:
                            display_tool_calls(tool_calls_container, _resp_chunk.tools)

                        # Display response
                        if _resp_chunk.content is not None:
                            response += _resp_chunk.content
                            resp_container.markdown(response)

                    add_message(
                        "assistant", response, agentic_rag_agent.run_response.tools
                    )
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    add_message("assistant", error_message)
                    st.error(error_message)

    ####################################################################
    # Session selector
    ####################################################################
    session_selector_widget(agentic_rag_agent, model_id)
    rename_session_widget(agentic_rag_agent)

    ####################################################################
    # About section
    ####################################################################
    about_widget()


main()
