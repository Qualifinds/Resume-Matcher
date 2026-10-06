import { describe, expect, it, vi } from 'vitest';
import { render } from '@testing-library/react';
import Resume, { type ResumeData } from '@/components/dashboard/resume-component';
import { cityCountryOnly, scrubContactText, toClientResume } from '@/lib/client-resume';
import type { TemplateType } from '@/lib/types/template-settings';

vi.mock('@/lib/i18n', () => ({ useTranslations: () => ({ t: (k: string) => k }) }));

const data = {
  personalInfo: {
    name: 'Ana Pérez',
    title: 'Backend Engineer',
    email: 'ana.perez@example.com',
    phone: '+52 55 1191 2660',
    location: 'Calle 5 #123, Col. Centro, Monterrey, NL, México',
    website: 'https://anaperez.dev',
    linkedin: 'linkedin.com/in/ana-perez',
    github: 'github.com/anaperez',
  },
  summary: 'Engineer with 8 years of experience. Reach me at ana.perez@example.com or 55 1191 2660.',
  workExperience: [
    {
      id: 1,
      title: 'Senior Engineer',
      company: 'Bimbo',
      location: 'Monterrey, México',
      years: '01-2019 - 03-2021',
      description: ['Led a team of 8 engineers.', 'See https://bimbo.example.com/ana for the case study.'],
    },
  ],
  education: [{ id: 1, institution: 'UANL', degree: 'BSc Computer Science', years: '2012-2016' }],
  personalProjects: [
    {
      id: 1,
      name: 'Open source tool',
      role: 'Author',
      years: '2022',
      github: 'https://github.com/anaperez/tool',
      website: 'https://tool.example.com',
      description: ['Used by 2k developers.'],
    },
  ],
  additional: { technicalSkills: ['Python', 'TypeScript'], languages: ['English', 'Spanish'] },
  customSections: {
    extra: { sectionType: 'text', text: 'Instagram @anaperez · ana@x.mx' },
  },
} as unknown as ResumeData;

describe('scrubContactText', () => {
  it('removes emails, URLs, handles and phones; keeps date ranges', () => {
    expect(scrubContactText('Mail ana@x.mx now')).not.toContain('@');
    expect(scrubContactText('See linkedin.com/in/ana-perez')).not.toContain('linkedin');
    expect(scrubContactText('IG @anaperez')).not.toContain('@anaperez');
    expect(scrubContactText('Tel 55 1191 2660')).not.toMatch(/\d{4}/);
    expect(scrubContactText('Bimbo (2019-2021)')).toBe('Bimbo (2019-2021)');
    expect(scrubContactText('Bimbo (01-2019 - 03-2021)')).toBe('Bimbo (01-2019 - 03-2021)');
    expect(scrubContactText('Led 8 engineers; cut costs by 40%.')).toBe('Led 8 engineers; cut costs by 40%.');
  });
});

describe('cityCountryOnly', () => {
  it('drops street, number and neighbourhood', () => {
    expect(cityCountryOnly('Calle 5 #123, Col. Centro, Monterrey, NL, México')).toBe('Monterrey, NL, México');
    expect(cityCountryOnly('Av. Insurgentes Sur 1234, CDMX, México')).toBe('CDMX, México');
    expect(cityCountryOnly('Monterrey, México')).toBe('Monterrey, México');
    expect(cityCountryOnly('64000')).toBeUndefined();
    expect(cityCountryOnly(undefined)).toBeUndefined();
  });
});

describe('toClientResume', () => {
  it('keeps name and experience, drops every contact field', () => {
    const out = toClientResume(data);
    expect(out.personalInfo).toEqual({
      name: 'Ana Pérez',
      title: 'Backend Engineer',
      location: 'Monterrey, NL, México',
    });
    expect(out.workExperience?.[0].company).toBe('Bimbo');
    expect(out.workExperience?.[0].years).toBe('01-2019 - 03-2021');
    expect(out.personalProjects?.[0].github).toBeUndefined();
    expect(out.personalProjects?.[0].website).toBeUndefined();
    expect(out.personalProjects?.[0].name).toBe('Open source tool');
    expect(JSON.stringify(out)).not.toMatch(/example\.com|@|linkedin|github|1191/);
  });

  it('does not mutate the original', () => {
    toClientResume(data);
    expect(data.personalInfo?.email).toBe('ana.perez@example.com');
  });
});

// Every template reads the same ResumeData, so the single cut-point must hold in all of them.
const TEMPLATES: TemplateType[] = [
  'swiss-single',
  'swiss-two-column',
  'modern',
  'modern-two-column',
  'latex',
  'clean',
  'vivid',
];

describe.each(TEMPLATES)('client variant renders no contact info — %s', (template) => {
  it('has the name and experience but no email, phone, link or handle', () => {
    const { container } = render(<Resume resumeData={toClientResume(data)} template={template} />);
    const html = container.innerHTML;
    // Some templates split the name across spans, so compare the visible text.
    expect(container.textContent).toContain('Ana Pérez');
    expect(container.textContent).toContain('Bimbo');
    expect(html).not.toMatch(/@/);
    expect(html).not.toMatch(/example\.com|linkedin|github|anaperez/i);
    expect(html).not.toMatch(/1191|2660/);
    expect(html).not.toMatch(/mailto:|tel:/i);
    expect(html).not.toMatch(/href=/i);
  });
});
