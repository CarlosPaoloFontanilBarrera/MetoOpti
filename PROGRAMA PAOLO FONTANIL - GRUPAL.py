def gj(m):
 n=len(m)
 for j in range(n):
  if m[j][j]==0:
   try:m[j],m[next(i for i in range(j+1,n)if m[i][j]!=0)]=m[next(i for i in range(j+1,n)if m[i][j]!=0)],m[j]
   except:raise ValueError("No solución única")
  m[j]=[x/m[j][j]for x in m[j]]
  for i in range(n):
   if i!=j:m[i]=[a-m[i][j]*b for a,b in zip(m[i],m[j])]
 return [round(r[-1],6)for r in m]

try:
 n=int(input("N incógnitas: "))
 m=[list(map(float,input(f"E{i+1}: ").split()))for i in range(n)]
 if any(len(r)!=n+1 for r in m):raise ValueError("Tamaño incorrecto")
 r=input("¿Restringir valores? (s/n): ").lower()=="s"
 rest=[tuple(map(float,input(f"x{i+1} min,max: ").split(",")))if r else None for i in range(n)]
 s=gj(m)
 print("Solución:",*[f"x{i+1}={v}"for i,v in enumerate(s)])
 if r:
  for i,(v,t)in enumerate(zip(s,rest)):
   if t and not(t[0]<=v<=t[1]):print(f"⚠️ x{i+1}={v} fuera de [{t[0]},{t[1]}]")
except Exception as e:print(f"Error: {e}")