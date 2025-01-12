from graphviz import Digraph

# Create a new UML diagram with enhanced visual settings
uml_diagram = Digraph('Enhanced_UML_Class_Diagram', filename='./Final_Enhanced_UML_Diagram.png', format='png')

# Global graph settings for better visualization
uml_diagram.attr(
    rankdir='TB',          # Top to bottom layout
    splines='ortho',       # Orthogonal lines
    nodesep='1.0',        # Increased node separation
    ranksep='1.2',        # Increased rank separation
    concentrate='true',    # Merge edges
    compound='true',       # Allow cluster edges
    fontname='Arial',      # Use Arial font
    dpi='300',            # High resolution
    pad='0.5'             # Add padding
)

# Define node attributes for cleaner look
uml_diagram.attr('node',
    shape='record',        # Use record shape for class boxes
    style='filled',       # Fill boxes
    fillcolor='white',    # White background
    fontname='Arial',     # Arial font
    fontsize='10',        # Readable font size
    margin='0.3,0.1',     # Margin for text
    height='0.5'          # Consistent height
)

# Define edge attributes for cleaner look
uml_diagram.attr('edge',
    fontname='Arial',     # Arial font
    fontsize='8',         # Smaller font for relationships
    len='1.5',           # Edge length
    splines='ortho'       # Orthogonal lines
)

# Create subgraphs for each layer for better organization
with uml_diagram.subgraph(name='cluster_0') as view_layer:
    view_layer.attr(label='Presentation Layer', style='rounded', color='#2C3E50')
    
    # View Classes
    view_layer.node('AppointmentView', '''AppointmentView|
- scanner: Scanner\\l
- appointmentController: AppointmentController\\l
- patientView: PatientView\\l
- dermatologistView: DermatologistView\\l
- treatmentView: TreatmentView\\l
- timeView: TimeView\\l
- invoiceView: InvoiceView\\l
|
+ showAppointmentMenu(): void\\l
+ displayMenu(): void\\l
+ getUserChoice(): int\\l
+ scheduleAppointment(): void\\l
+ viewAllAppointments(): void\\l
+ updateAppointment(): void\\l
+ cancelAppointment(): void\\l
+ completeAppointment(): void\\l
+ processInvoicePayment(): void\\l
+ searchAppointment(): void\\l''')

    view_layer.node('PatientView', '''PatientView|
|
+ getPatientDetails(): Patient\\l''')

    view_layer.node('DermatologistView', '''DermatologistView|
- scanner: Scanner\\l
- dermatologistService: DermatologistService\\l
|
+ selectDermatologist(): DermatologistModel\\l
+ addTimeSlotForDermatologist(): void\\l
+ getValidDateFromUser(): LocalDate\\l
+ getValidStartTimeFromUser(): LocalTime\\l''')

    view_layer.node('TreatmentView', '''TreatmentView|
|
+ getTreatmentDetails(): Treatment\\l''')

    view_layer.node('TimeView', '''TimeView|
|
+ getTimeDetails(dermatologist: DermatologistModel): Time\\l''')

    view_layer.node('InvoiceView', '''InvoiceView|
|
+ getInvoiceDetails(treatment: Treatment): Invoice\\l''')

with uml_diagram.subgraph(name='cluster_1') as service_layer:
    service_layer.attr(label='Service Layer', style='rounded', color='#2980B9')
    
    # Service Classes
    service_layer.node('AppointmentService', '''AppointmentService|
- appointmentRepository: AppointmentRepository\\l
|
+ createAppointment(appointment: Appointment): void\\l
+ listAppointments(): List<Appointment>\\l
+ updateAppointment(appointment: Appointment): boolean\\l
+ cancelAppointment(appointmentId: int): boolean\\l
+ completeAppointment(appointmentId: int): boolean\\l
+ searchAppointments(criteria: String): List<Appointment>\\l''')

    service_layer.node('DermatologistService', '''DermatologistService|
- dermatologistRepository: DermatologistRepository\\l
|
+ selectDermatologist(): DermatologistModel\\l
+ addTimeSlotForDermatologist(dermatologist: DermatologistModel, time: Time): boolean\\l
+ getDermatologistAvailability(dermatologistId: int, date: LocalDate): List<Time>\\l''')

    service_layer.node('PatientService', '''PatientService|
- patientRepository: PatientRepository\\l
|
+ getPatientDetails(patientId: int): Patient\\l
+ createPatient(patient: Patient): void\\l
+ updatePatient(patient: Patient): boolean\\l''')

with uml_diagram.subgraph(name='cluster_2') as repository_layer:
    repository_layer.attr(label='Repository Layer', style='rounded', color='#27AE60')
    
    # Repository Classes
    repository_layer.node('AppointmentRepository', '''AppointmentRepository|
- appointments: List<Appointment>\\l
|
+ save(appointment: Appointment): void\\l
+ findAll(): List<Appointment>\\l
+ findById(appointmentId: int): Optional<Appointment>\\l
+ delete(appointmentId: int): boolean\\l
+ update(appointment: Appointment): boolean\\l''')

    repository_layer.node('DermatologistRepository', '''DermatologistRepository|
- dermatologists: List<DermatologistModel>\\l
|
+ save(dermatologist: DermatologistModel): void\\l
+ findAll(): List<DermatologistModel>\\l
+ findById(id: int): Optional<DermatologistModel>\\l
+ addTimeSlot(dermatologist: DermatologistModel, time: Time): boolean\\l''')

    repository_layer.node('PatientRepository', '''PatientRepository|
- patients: List<Patient>\\l
|
+ save(patient: Patient): void\\l
+ findAll(): List<Patient>\\l
+ findById(id: int): Optional<Patient>\\l''')

with uml_diagram.subgraph(name='cluster_3') as domain_layer:
    domain_layer.attr(label='Domain Layer', style='rounded', color='#8E44AD')
    
    # Domain Model Classes
    domain_layer.node('Appointment', '''Appointment|
- appointmentId: int\\l
- patient: Patient\\l
- dermatologist: DermatologistModel\\l
- treatment: Treatment\\l
- time: Time\\l
- invoice: Invoice\\l
- status: AppointmentStatus\\l
|
+ setStatus(status: AppointmentStatus): void\\l
+ getInvoice(): Invoice\\l
+ getId(): int\\l''')

    domain_layer.node('Patient', '''Patient|
- patientId: int\\l
- name: String\\l
- age: int\\l
- contactInfo: String\\l
- medicalHistory: String\\l
|
+ getId(): int\\l
+ getName(): String\\l
+ getAge(): int\\l''')

    domain_layer.node('DermatologistModel', '''DermatologistModel|
- dermatologistId: int\\l
- name: String\\l
- specialization: String\\l
- availableTimeSlots: List<Time>\\l
|
+ checkAvailability(time: Time): boolean\\l
+ addTimeSlot(time: Time): void\\l''')

    domain_layer.node('Treatment', '''Treatment|
- treatmentId: int\\l
- name: String\\l
- description: String\\l
- duration: int\\l
- cost: double\\l
|
+ getCost(): double\\l
+ getDuration(): int\\l''')

    domain_layer.node('Invoice', '''Invoice|
- invoiceId: int\\l
- totalAmount: double\\l
- amountPaid: double\\l
- status: PaymentStatus\\l
|
+ makePayment(amount: double): void\\l
+ isPaid(): boolean\\l
+ getBalance(): double\\l''')

    domain_layer.node('Time', '''Time|
- date: LocalDate\\l
- startTime: LocalTime\\l
- endTime: LocalTime\\l
|
+ getDate(): LocalDate\\l
+ getStartTime(): LocalTime\\l
+ getEndTime(): LocalTime\\l
+ isAvailable(): boolean\\l''')

# Add relationships with better edge styling
# View to Service Layer relationships
uml_diagram.edge('AppointmentView', 'AppointmentService', 'uses', dir='forward', arrowhead='vee')
uml_diagram.edge('DermatologistView', 'DermatologistService', 'uses', dir='forward', arrowhead='vee')
uml_diagram.edge('PatientView', 'PatientService', 'uses', dir='forward', arrowhead='vee')

# Service to Repository Layer relationships
uml_diagram.edge('AppointmentService', 'AppointmentRepository', 'uses', dir='forward', arrowhead='vee')
uml_diagram.edge('DermatologistService', 'DermatologistRepository', 'uses', dir='forward', arrowhead='vee')
uml_diagram.edge('PatientService', 'PatientRepository', 'uses', dir='forward', arrowhead='vee')

# Domain Model relationships
uml_diagram.edge('Appointment', 'Patient', 'contains', dir='both', arrowtail='odiamond')
uml_diagram.edge('Appointment', 'DermatologistModel', 'contains', dir='both', arrowtail='odiamond')
uml_diagram.edge('Appointment', 'Treatment', 'contains', dir='both', arrowtail='odiamond')
uml_diagram.edge('Appointment', 'Time', 'contains', dir='both', arrowtail='odiamond')
uml_diagram.edge('Appointment', 'Invoice', 'owns', dir='both', arrowtail='diamond')

# Repository to Domain Model relationships
uml_diagram.edge('AppointmentRepository', 'Appointment', 'manages', style='dashed', dir='forward')
uml_diagram.edge('DermatologistRepository', 'DermatologistModel', 'manages', style='dashed', dir='forward')
uml_diagram.edge('PatientRepository', 'Patient', 'manages', style='dashed', dir='forward')

# Generate the diagram
uml_diagram.render(view=True)
