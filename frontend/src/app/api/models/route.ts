import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8080';
const BACKEND_API_KEY = process.env.BACKEND_API_KEY;

export async function GET(request: NextRequest) {
  if (!BACKEND_API_KEY) {
    return NextResponse.json(
      { error: 'Backend API key not configured' },
      { status: 500 }
    );
  }

  try {
    const response = await fetch(`${BACKEND_URL}/models`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${BACKEND_API_KEY}`,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching models:', error);
    return NextResponse.json(
      { error: 'Failed to fetch models from backend' },
      { status: 500 }
    );
  }
}
