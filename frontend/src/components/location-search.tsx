"use client"

import { useState, useEffect, useRef } from "react"
import { Search, Loader2, MapPin } from 'lucide-react'
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { useDebounce } from "@/hooks/use-debounce"

interface LocationSearchProps {
  onSelect: (name: string, lat: number, lon: number) => void
}

interface LocationResult {
  place_id: number
  display_name: string
  lat: string
  lon: string
}

interface ThaiProvince {
  name: string
  display_name: string
  lat: number
  lon: number
}

// List of Thai provinces with coordinates
const thaiProvinces: ThaiProvince[] = [
  { name: "กรุงเทพมหานคร", display_name: "กรุงเทพมหานคร, ประเทศไทย", lat: 13.7563, lon: 100.5018 },
  { name: "สมุทรปราการ", display_name: "จังหวัดสมุทรปราการ, ประเทศไทย", lat: 13.5991, lon: 100.5998 },
  { name: "นนทบุรี", display_name: "จังหวัดนนทบุรี, ประเทศไทย", lat: 13.8622, lon: 100.5134 },
  { name: "ปทุมธานี", display_name: "จังหวัดปทุมธานี, ประเทศไทย", lat: 14.0208, lon: 100.5255 },
  { name: "พระนครศรีอยุธยา", display_name: "จังหวัดพระนครศรีอยุธยา, ประเทศไทย", lat: 14.3692, lon: 100.5876 },
  { name: "อ่างทอง", display_name: "จังหวัดอ่างทอง, ประเทศไทย", lat: 14.5896, lon: 100.4546 },
  { name: "ลพบุรี", display_name: "จังหวัดลพบุรี, ประเทศไทย", lat: 14.7995, lon: 100.6534 },
  { name: "สิงห์บุรี", display_name: "จังหวัดสิงห์บุรี, ประเทศไทย", lat: 14.8911, lon: 100.3965 },
  { name: "ชัยนาท", display_name: "จังหวัดชัยนาท, ประเทศไทย", lat: 15.1851, lon: 100.1251 },
  { name: "สระบุรี", display_name: "จังหวัดสระบุรี, ประเทศไทย", lat: 14.5289, lon: 100.9108 },
  { name: "ชลบุรี", display_name: "จังหวัดชลบุรี, ประเทศไทย", lat: 13.3611, lon: 100.9847 },
  { name: "ระยอง", display_name: "จังหวัดระยอง, ประเทศไทย", lat: 12.6833, lon: 101.2372 },
  { name: "จันทบุรี", display_name: "จังหวัดจันทบุรี, ประเทศไทย", lat: 12.6111, lon: 102.1035 },
  { name: "ตราด", display_name: "จังหวัดตราด, ประเทศไทย", lat: 12.2427, lon: 102.5177 },
  { name: "ฉะเชิงเทรา", display_name: "จังหวัดฉะเชิงเทรา, ประเทศไทย", lat: 13.6904, lon: 101.0779 },
  { name: "ปราจีนบุรี", display_name: "จังหวัดปราจีนบุรี, ประเทศไทย", lat: 14.0579, lon: 101.3736 },
  { name: "นครนายก", display_name: "จังหวัดนครนายก, ประเทศไทย", lat: 14.2069, lon: 101.2131 },
  { name: "สระแก้ว", display_name: "จังหวัดสระแก้ว, ประเทศไทย", lat: 13.8241, lon: 102.0645 },
  { name: "นครราชสีมา", display_name: "จังหวัดนครราชสีมา, ประเทศไทย", lat: 14.9798, lon: 102.0978 },
  { name: "บุรีรัมย์", display_name: "จังหวัดบุรีรัมย์, ประเทศไทย", lat: 14.9954, lon: 103.1059 },
  { name: "สุรินทร์", display_name: "จังหวัดสุรินทร์, ประเทศไทย", lat: 14.8813, lon: 103.4960 },
  { name: "ศรีสะเกษ", display_name: "จังหวัดศรีสะเกษ, ประเทศไทย", lat: 15.1186, lon: 104.3218 },
  { name: "อุบลราชธานี", display_name: "จังหวัดอุบลราชธานี, ประเทศไทย", lat: 15.2286, lon: 104.9007 },
  { name: "ยโสธร", display_name: "จังหวัดยโสธร, ประเทศไทย", lat: 15.7923, lon: 104.1452 },
  { name: "ชัยภูมิ", display_name: "จังหวัดชัยภูมิ, ประเทศไทย", lat: 15.8068, lon: 102.0317 },
  { name: "อำนาจเจริญ", display_name: "จังหวัดอำนาจเจริญ, ประเทศไทย", lat: 15.8656, lon: 104.6266 },
  { name: "หนองบัวลำภู", display_name: "จังหวัดหนองบัวลำภู, ประเทศไทย", lat: 17.2217, lon: 102.4266 },
  { name: "ขอนแก่น", display_name: "จังหวัดขอนแก่น, ประเทศไทย", lat: 16.4419, lon: 102.8360 },
  { name: "อุดรธานี", display_name: "จังหวัดอุดรธานี, ประเทศไทย", lat: 17.4139, lon: 102.7871 },
  { name: "เลย", display_name: "จังหวัดเลย, ประเทศไทย", lat: 17.4860, lon: 101.7223 },
  { name: "หนองคาย", display_name: "จังหวัดหนองคาย, ประเทศไทย", lat: 17.8782, lon: 102.7418 },
  { name: "มหาสารคาม", display_name: "จังหวัดมหาสารคาม, ประเทศไทย", lat: 16.1851, lon: 103.3029 },
  { name: "ร้อยเอ็ด", display_name: "จังหวัดร้อยเอ็ด, ประเทศไทย", lat: 16.0538, lon: 103.6520 },
  { name: "กาฬสินธุ์", display_name: "จังหวัดกาฬสินธุ์, ประเทศไทย", lat: 16.4314, lon: 103.5058 },
  { name: "สกลนคร", display_name: "จังหวัดสกลนคร, ประเทศไทย", lat: 17.1664, lon: 104.1486 },
  { name: "นครพนม", display_name: "จังหวัดนครพนม, ประเทศไทย", lat: 17.4048, lon: 104.7690 },
  { name: "มุกดาหาร", display_name: "จังหวัดมุกดาหาร, ประเทศไทย", lat: 16.5425, lon: 104.7227 },
  { name: "เชียงใหม่", display_name: "จังหวัดเชียงใหม่, ประเทศไทย", lat: 18.7883, lon: 98.9853 },
  { name: "ลำพูน", display_name: "จังหวัดลำพูน, ประเทศไทย", lat: 18.5747, lon: 99.0087 },
  { name: "ลำปาง", display_name: "จังหวัดลำปาง, ประเทศไทย", lat: 18.2783, lon: 99.5080 },
  { name: "อุตรดิตถ์", display_name: "จังหวัดอุตรดิตถ์, ประเทศไทย", lat: 17.6200, lon: 100.0993 },
  { name: "แพร่", display_name: "จังหวัดแพร่, ประเทศไทย", lat: 18.1445, lon: 100.1398 },
  { name: "น่าน", display_name: "จังหวัดน่าน, ประเทศไทย", lat: 18.7756, lon: 100.7730 },
  { name: "พะเยา", display_name: "จังหวัดพะเยา, ประเทศไทย", lat: 19.1664, lon: 99.9003 },
  { name: "เชียงราย", display_name: "จังหวัดเชียงราย, ประเทศไทย", lat: 19.9071, lon: 99.8305 },
  { name: "แม่ฮ่องสอน", display_name: "จังหวัดแม่ฮ่องสอน, ประเทศไทย", lat: 19.3020, lon: 97.9654 },
  { name: "นครสวรรค์", display_name: "จังหวัดนครสวรรค์, ประเทศไทย", lat: 15.7030, lon: 100.1367 },
  { name: "อุทัยธานี", display_name: "จังหวัดอุทัยธานี, ประเทศไทย", lat: 15.3835, lon: 100.0255 },
  { name: "กำแพงเพชร", display_name: "จังหวัดกำแพงเพชร, ประเทศไทย", lat: 16.4831, lon: 99.5263 },
  { name: "ตาก", display_name: "จังหวัดตาก, ประเทศไทย", lat: 16.8839, lon: 99.1258 },
  { name: "สุโขทัย", display_name: "จังหวัดสุโขทัย, ประเทศไทย", lat: 17.0100, lon: 99.8265 },
  { name: "พิษณุโลก", display_name: "จังหวัดพิษณุโลก, ประเทศไทย", lat: 16.8211, lon: 100.2659 },
  { name: "พิจิตร", display_name: "จังหวัดพิจิตร, ประเทศไทย", lat: 16.4429, lon: 100.3487 },
  { name: "เพชรบูรณ์", display_name: "จังหวัดเพชรบูรณ์, ประเทศไทย", lat: 16.4189, lon: 101.1591 },
  { name: "ราชบุรี", display_name: "จังหวัดราชบุรี, ประเทศไทย", lat: 13.5282, lon: 99.8133 },
  { name: "กาญจนบุรี", display_name: "จังหวัดกาญจนบุรี, ประเทศไทย", lat: 14.0227, lon: 99.5328 },
  { name: "สุพรรณบุรี", display_name: "จังหวัดสุพรรณบุรี, ประเทศไทย", lat: 14.4744, lon: 100.1177 },
  { name: "นครปฐม", display_name: "จังหวัดนครปฐม, ประเทศไทย", lat: 13.8196, lon: 100.0655 },
  { name: "สมุทรสาคร", display_name: "จังหวัดสมุทรสาคร, ประเทศไทย", lat: 13.5475, lon: 100.2747 },
  { name: "สมุทรสงคราม", display_name: "จังหวัดสมุทรสงคราม, ประเทศไทย", lat: 13.4098, lon: 100.0021 },
  { name: "เพชรบุรี", display_name: "จังหวัดเพชรบุรี, ประเทศไทย", lat: 13.1119, lon: 99.9406 },
  { name: "ประจวบคีรีขันธ์", display_name: "จังหวัดประจวบคีรีขันธ์, ประเทศไทย", lat: 11.8126, lon: 99.7957 },
  { name: "นครศรีธรรมราช", display_name: "จังหวัดนครศรีธรรมราช, ประเทศไทย", lat: 8.4304, lon: 99.9633 },
  { name: "กระบี่", display_name: "จังหวัดกระบี่, ประเทศไทย", lat: 8.0862, lon: 98.9062 },
  { name: "พังงา", display_name: "จังหวัดพังงา, ประเทศไทย", lat: 8.4510, lon: 98.5314 },
  { name: "ภูเก็ต", display_name: "จังหวัดภูเก็ต, ประเทศไทย", lat: 7.9519, lon: 98.3381 },
  { name: "สุราษฎร์ธานี", display_name: "จังหวัดสุราษฎร์ธานี, ประเทศไทย", lat: 9.1383, lon: 99.3217 },
  { name: "ระนอง", display_name: "จังหวัดระนอง, ประเทศไทย", lat: 9.9528, lon: 98.6084 },
  { name: "ชุมพร", display_name: "จังหวัดชุมพร, ประเทศไทย", lat: 10.4930, lon: 99.1800 },
  { name: "สงขลา", display_name: "จังหวัดสงขลา, ประเทศไทย", lat: 7.1756, lon: 100.6142 },
  { name: "สตูล", display_name: "จังหวัดสตูล, ประเทศไทย", lat: 6.6238, lon: 100.0678 },
  { name: "ตรัง", display_name: "จังหวัดตรัง, ประเทศไทย", lat: 7.5593, lon: 99.6113 },
  { name: "พัทลุง", display_name: "จังหวัดพัทลุง, ประเทศไทย", lat: 7.6166, lon: 100.0742 },
  { name: "ปัตตานี", display_name: "จังหวัดปัตตานี, ประเทศไทย", lat: 6.8692, lon: 101.2550 },
  { name: "ยะลา", display_name: "จังหวัดยะลา, ประเทศไทย", lat: 6.5413, lon: 101.2803 },
  { name: "นราธิวาส", display_name: "จังหวัดนราธิวาส, ประเทศไทย", lat: 6.4251, lon: 101.8257 },
  { name: "บึงกาฬ", display_name: "จังหวัดบึงกาฬ, ประเทศไทย", lat: 18.3609, lon: 103.6466 },
];

export default function LocationSearch({ onSelect }: LocationSearchProps) {
  const [open, setOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")
  const debouncedSearchTerm = useDebounce(searchTerm, 500)
  const [results, setResults] = useState<LocationResult[]>([])
  const [provinceResults, setProvinceResults] = useState<ThaiProvince[]>([])
  const [loading, setLoading] = useState(false)
  const [inputValue, setInputValue] = useState("")
  const inputRef = useRef<HTMLInputElement>(null)

  // Function to match Thai provinces based on partial input
  const matchThaiProvinces = (input: string): ThaiProvince[] => {
    if (!input || input.length < 2) return []
    
    // Create a regex that will match the input anywhere in the province name
    // Using case-insensitive matching for Latin characters
    const regex = new RegExp(input.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')
    
    return thaiProvinces.filter(province => 
      // Match against the province name
      regex.test(province.name) || 
      // Also match against the display name (which includes จังหวัด prefix)
      regex.test(province.display_name)
    )
  }

  useEffect(() => {
    if (debouncedSearchTerm.length < 2) {
      setResults([])
      setProvinceResults([])
      return
    }

    // First, check for matching Thai provinces
    const matchedProvinces = matchThaiProvinces(debouncedSearchTerm)
    setProvinceResults(matchedProvinces)

    // Then, search using Nominatim API
    const searchLocations = async () => {
      setLoading(true)
      try {
        // Nominatim API with focus on Thailand
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(
            debouncedSearchTerm,
          )}+Thailand&format=json&addressdetails=1&limit=10&countrycodes=th`,
        )

        if (!response.ok) {
          throw new Error("Network response was not ok")
        }

        const data = await response.json()
        
        // Filter out results that are already in the province results
        // to avoid duplicates
        const provinceNames = matchedProvinces.map(p => p.display_name)
        const filteredResults = data.filter((result: LocationResult) => 
          !provinceNames.some(name => result.display_name.includes(name))
        )
        
        setResults(filteredResults)
      } catch (error) {
        console.error("Error fetching locations:", error)
        setResults([])
      } finally {
        setLoading(false)
      }
    }

    searchLocations()
  }, [debouncedSearchTerm])

  const handleSelectProvince = (province: ThaiProvince) => {
    setInputValue(province.display_name)
    onSelect(province.display_name, province.lat, province.lon)
    setOpen(false)
  }

  const handleSelectResult = (result: LocationResult) => {
    setInputValue(result.display_name)
    onSelect(result.display_name, Number.parseFloat(result.lat), Number.parseFloat(result.lon))
    setOpen(false)
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <div className="relative">
          <Input
            ref={inputRef}
            placeholder="Search for a location in Thailand..."
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value)
              setSearchTerm(e.target.value)
            }}
            onClick={() => setOpen(true)}
            className="w-full pr-10"
          />
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-0 top-0 h-full px-3 text-gray-400 hover:text-gray-600"
            onClick={() => {
              setOpen(true)
              inputRef.current?.focus()
            }}
          >
            <Search className="h-4 w-4" />
          </Button>
        </div>
      </PopoverTrigger>
      <PopoverContent className="p-0 w-[var(--radix-popover-trigger-width)]" align="start">
        <Command>
          <CommandInput
            placeholder="Search locations..."
            value={searchTerm}
            onValueChange={setSearchTerm}
            className="h-9"
          />
          <CommandList>
            {loading && provinceResults.length === 0 && results.length === 0 && (
              <div className="flex items-center justify-center py-6">
                <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
              </div>
            )}
            
            {!loading && provinceResults.length === 0 && results.length === 0 && (
              <CommandEmpty>
                {debouncedSearchTerm.length < 2 ? "Type at least 2 characters to search" : "No locations found"}
              </CommandEmpty>
            )}
            
            {provinceResults.length > 0 && (
              <CommandGroup heading="Provinces">
                {provinceResults.map((province) => (
                  <CommandItem 
                    key={province.name} 
                    value={province.display_name} 
                    onSelect={() => handleSelectProvince(province)}
                  >
                    <MapPin className="mr-2 h-4 w-4 text-blue-600" />
                    <div className="flex flex-col">
                      <span className="font-medium">{province.display_name}</span>
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
            
            {results.length > 0 && (
              <CommandGroup heading="Other Locations">
                {results.map((result) => (
                  <CommandItem 
                    key={result.place_id} 
                    value={result.display_name} 
                    onSelect={() => handleSelectResult(result)}
                  >
                    <div className="flex flex-col">
                      <span className="font-medium">{result.display_name.split(", ").slice(0, 2).join(", ")}</span>
                      <span className="text-xs text-gray-500 truncate">
                        {result.display_name.split(", ").slice(2).join(", ")}
                      </span>
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
