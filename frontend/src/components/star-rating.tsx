// components/star-rating.tsx
"use client"

import { useState } from "react"
import { Star } from "lucide-react"
import { toast } from "sonner"

interface StarRatingProps {
  attractionName: string
  latitude: number
  longitude: number
  onRatingSubmit?: (rating: number) => void
}

export default function StarRating({ attractionName, latitude, longitude, onRatingSubmit }: StarRatingProps) {
  const [rating, setRating] = useState<number>(0)
  const [hoveredRating, setHoveredRating] = useState<number>(0)
  const [submitted, setSubmitted] = useState<boolean>(false)
  const [submitting, setSubmitting] = useState<boolean>(false)

  const handleRatingClick = async (selectedRating: number) => {
    if (submitted) return
    
    setRating(selectedRating)
    setSubmitting(true)
    
    try {
      const response = await fetch("http://localhost:5004/api/rate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ATT_NAME_TH: attractionName,
          ATT_LATITUDE: latitude,
          ATT_LONGITUDE: longitude,
          rating: selectedRating,
        }),
      })

      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || "Failed to submit rating")
      }
      
      setSubmitted(true)
      if (onRatingSubmit) onRatingSubmit(selectedRating)
      toast.success("Thank you for your rating!")
    } catch (error) {
      console.error("Error submitting rating:", error)
      toast.error("Failed to submit rating. Please try again.")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex flex-col items-center mt-2">
      <p className="text-sm text-gray-600 mb-1">
        {submitted ? "Your rating:" : "Rate this attraction:"}
      </p>
      <div className="flex items-center">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            type="button"
            disabled={submitted || submitting}
            className={`p-1 focus:outline-none transition-colors ${
              submitted ? "cursor-default" : "cursor-pointer"
            }`}
            onMouseEnter={() => !submitted && setHoveredRating(star)}
            onMouseLeave={() => !submitted && setHoveredRating(0)}
            onClick={() => handleRatingClick(star)}
            aria-label={`Rate ${star} stars`}
          >
            <Star
              className={`h-5 w-5 ${
                star <= (hoveredRating || rating)
                  ? "fill-yellow-400 text-yellow-400"
                  : "text-gray-300"
              } ${submitting ? "opacity-50" : ""}`}
            />
          </button>
        ))}
        {submitting && (
          <div className="ml-2 h-4 w-4 animate-spin rounded-full border-2 border-yellow-400 border-t-transparent"></div>
        )}
      </div>
    </div>
  )
}